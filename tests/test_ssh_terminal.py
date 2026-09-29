import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from opkit.errors import ProtocolError
from opkit.protocols import ssh_terminal as ssh


class SSHTerminalLifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.manager = ssh.SSHTerminalManager()
        self.manager._protocol_of = lambda config, device: SimpleNamespace(
            encoding="utf-8"
        )
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.created = []

        async def connect(session):
            self.created.append(session)
            self.started.set()
            await self.release.wait()
            session.process = SimpleNamespace(close=Mock())

        self.connect_patch = patch.object(ssh.SSHTerminalSession, "connect", connect)
        self.connect_patch.start()
        self.addCleanup(self.connect_patch.stop)
        self.addAsyncCleanup(self.manager.close_all)

    def open(self, device="switch"):
        return asyncio.create_task(
            self.manager.open(None, device, quiet_timeout_ms=0, deadline_ms=0)
        )

    async def wait(self, event):
        await asyncio.wait_for(event.wait(), timeout=2)

    async def test_reaper_does_not_close_pending_connection(self):
        tick = asyncio.Event()
        sentinel = ssh.SSHTerminalSession("ready", SimpleNamespace(encoding="utf-8"))
        sentinel.process = SimpleNamespace(close=Mock())
        sentinel.expired = lambda now: tick.set() or False
        self.manager._sessions["ready"] = sentinel
        with patch.object(ssh, "REAP_INTERVAL_SECONDS", 0.001):
            self.manager._ensure_reaper()
            opening = self.open()
            await self.wait(self.started)
            await self.wait(tick)
            self.assertFalse(self.created[0]._transcript.closed)
            self.assertTrue(await self.manager.occupied("switch"))
            self.release.set()
            session, result = await asyncio.wait_for(opening, 2)
            self.assertTrue(session.connected)
            self.assertEqual(result.connection_state, "open")
            self.assertIs(await self.manager.get("switch"), session)

    async def test_concurrent_opens_share_connection(self):
        first = self.open()
        await self.wait(self.started)
        second = self.open()
        await asyncio.sleep(0)
        self.release.set()
        a, b = await asyncio.wait_for(asyncio.gather(first, second), 2)
        self.assertIs(a[0], b[0])
        self.assertEqual(len(self.created), 1)

    async def test_cancelled_waiter_does_not_cancel_shared_connection(self):
        first = self.open()
        await self.wait(self.started)
        second = self.open()
        await asyncio.sleep(0)
        first.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await first
        self.release.set()
        session, _ = await asyncio.wait_for(second, 2)
        self.assertTrue(session.connected)
        self.assertEqual(len(self.created), 1)

    async def test_close_cancels_pending_connection_and_allows_reopen(self):
        opening = self.open()
        await self.wait(self.started)
        await self.manager.close("switch")
        with self.assertRaises(asyncio.CancelledError):
            await opening
        self.assertTrue(self.created[0]._transcript.closed)
        self.assertFalse(await self.manager.occupied("switch"))
        self.release.set()
        session, _ = await asyncio.wait_for(self.open(), 2)
        self.assertTrue(session.connected)

    async def test_close_before_replacement_task_starts_closes_stale_session(self):
        stale = ssh.SSHTerminalSession("switch", SimpleNamespace(encoding="utf-8"))
        self.manager._sessions["switch"] = stale
        opening = self.open()
        # Let open() reserve a task, but close before that task starts.
        await asyncio.sleep(0)
        await self.manager.close("switch")
        with self.assertRaises(asyncio.CancelledError):
            await opening
        self.assertTrue(stale._transcript.closed)
        self.assertFalse(await self.manager.occupied("switch"))

    async def test_close_all_cancels_pending_connections(self):
        opening = self.open()
        await self.wait(self.started)
        self.assertEqual(await self.manager.close_all(), 1)
        with self.assertRaises(asyncio.CancelledError):
            await opening
        self.assertTrue(self.created[0]._transcript.closed)
        self.assertEqual(await self.manager.list(), [])
        self.assertFalse(await self.manager.occupied("switch"))
        self.assertIsNone(self.manager._reaper_task)

    async def test_failed_connection_releases_resources_and_capacity(self):
        sessions = []

        async def fail(session):
            sessions.append(session)
            raise OSError("handshake failed")

        with patch.object(ssh.SSHTerminalSession, "connect", fail):
            with self.assertRaisesRegex(OSError, "handshake failed"):
                await self.open()
        self.assertTrue(sessions[0]._transcript.closed)
        self.assertFalse(await self.manager.occupied("switch"))
        self.release.set()
        with patch.object(ssh, "MAX_SESSIONS", 1):
            session, _ = await asyncio.wait_for(self.open(), 2)
        self.assertTrue(session.connected)

    async def test_pending_connections_count_toward_limit(self):
        with patch.object(ssh, "MAX_SESSIONS", 1):
            first = self.open()
            await self.wait(self.started)
            with self.assertRaisesRegex(ProtocolError, "maximum number"):
                await self.open("another-switch")
            self.release.set()
            await asyncio.wait_for(first, 2)

    async def test_reaper_does_not_remove_replacement_session(self):
        self.release.set()
        old, _ = await self.open()
        closing = asyncio.Event()
        finish_close = asyncio.Event()
        cleaned = asyncio.Event()
        original_close = old.close
        calls = 0

        async def delayed_close():
            nonlocal calls
            calls += 1
            if calls == 1:
                await original_close()
                closing.set()
                await finish_close.wait()
                cleaned.set()
            else:
                await original_close()

        old.close = delayed_close
        old.expired = lambda now: True
        with patch.object(ssh, "REAP_INTERVAL_SECONDS", 0.001):
            # Restart the reaper so this test controls its next interval.
            self.manager._reaper_task.cancel()
            await asyncio.gather(self.manager._reaper_task, return_exceptions=True)
            self.manager._reaper_task = None
            self.manager._ensure_reaper()
            await self.wait(closing)
            replacement, _ = await asyncio.wait_for(self.open(), 2)
            finish_close.set()
            await self.wait(cleaned)
            self.assertIs(await self.manager.get("switch"), replacement)
            self.assertTrue(replacement.connected)

    async def test_reaper_still_removes_expired_established_session(self):
        self.release.set()
        session, _ = await self.open()
        removed = asyncio.Event()
        original_close = session.close

        async def close():
            await original_close()
            removed.set()

        session.close = close
        session.expired = lambda now: True
        with patch.object(ssh, "REAP_INTERVAL_SECONDS", 0.001):
            self.manager._reaper_task.cancel()
            await asyncio.gather(self.manager._reaper_task, return_exceptions=True)
            self.manager._reaper_task = None
            self.manager._ensure_reaper()
            await self.wait(removed)
            self.assertTrue(session._transcript.closed)
            self.assertEqual(await self.manager.list(), [])


if __name__ == "__main__":
    unittest.main()
