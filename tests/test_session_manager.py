import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock

from src.session_manager import (
    DEFAULT_CZECH_SESSION_ID,
    SessionManager,
)


class TestSessionManager(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = unittest.mock.patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"})
        self._tmp.start()
        import tempfile

        self.temp_dir = tempfile.TemporaryDirectory()
        self.sessions_dir = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()
        self._tmp.stop()

    def _make_mock_client(self, responses):
        client = MagicMock()
        side_effects = [SimpleNamespace(text=r) for r in responses]
        client.models.generate_content.side_effect = side_effects
        return client

    def test_default_session_created_and_persists_across_restarts(self) -> None:
        mgr1 = SessionManager(sessions_dir=self.sessions_dir, client=MagicMock())
        active = mgr1.get_active_session()
        self.assertEqual(active.session_id, DEFAULT_CZECH_SESSION_ID)

        created = mgr1.create_session(title="Покупки", description="Интернет-магазины")
        mgr1.rename_session(created.session_id, "Покупки в интернете")
        created.add_message("user", "Привет")
        mgr1._save_session(created)
        mgr1.set_pinned(True)

        # Simulate PC reboot by instantiating a fresh SessionManager from disk
        mgr2 = SessionManager(sessions_dir=self.sessions_dir, client=MagicMock())
        self.assertTrue(mgr2.pinned)
        self.assertEqual(mgr2.get_active_session().session_id, created.session_id)
        self.assertEqual(mgr2.get_active_session().title, "Покупки в интернете")
        self.assertEqual(len(mgr2.get_active_session().history), 1)

    def test_route_message_create_and_switch(self) -> None:
        router_create = json.dumps(
            {
                "action": "create",
                "target_session_id": "",
                "new_title": "Покупки в интернете",
                "new_description": "Выбор товаров и онлайн-заказы",
            }
        )
        client = self._make_mock_client([router_create])
        mgr = SessionManager(sessions_dir=self.sessions_dir, client=client)

        decision = mgr.route_message("Хочу заказать новый ноутбук онлайн")
        self.assertEqual(decision.action, "create")
        self.assertTrue(decision.created)
        self.assertEqual(decision.title, "Покупки в интернете")
        self.assertEqual(mgr.get_active_session().session_id, decision.session_id)

        # Now route back to czech_learning
        router_switch = json.dumps(
            {
                "action": "switch",
                "target_session_id": DEFAULT_CZECH_SESSION_ID,
                "new_title": "",
                "new_description": "",
            }
        )
        mgr._client = self._make_mock_client([router_switch])
        decision2 = mgr.route_message("Как перевести слово 'mluvit'?")
        self.assertEqual(decision2.action, "switch")
        self.assertTrue(decision2.switched)
        self.assertEqual(decision2.session_id, DEFAULT_CZECH_SESSION_ID)

    def test_route_message_respects_pinned_and_handles_router_error(self) -> None:
        client = MagicMock()
        client.models.generate_content.side_effect = RuntimeError("network down")
        mgr = SessionManager(sessions_dir=self.sessions_dir, client=client)

        # Fallback on error keeps active session
        decision = mgr.route_message("Что-то пошло не так")
        self.assertEqual(decision.action, "continue")
        self.assertEqual(decision.reason, "router_fallback")

        # Pinned skips client call completely
        client.models.generate_content.reset_mock()
        mgr.set_pinned(True)
        decision_pinned = mgr.route_message("Новая тема")
        self.assertEqual(decision_pinned.reason, "pinned")
        client.models.generate_content.assert_not_called()

    def test_generate_reply_saves_history(self) -> None:
        client = self._make_mock_client(["Ahoj! Jak se máš?"])
        mgr = SessionManager(sessions_dir=self.sessions_dir, client=client)

        reply = mgr.generate_reply("Скажи привет по-чешски")
        self.assertEqual(reply, "Ahoj! Jak se máš?")

        reloaded = SessionManager(sessions_dir=self.sessions_dir, client=client)
        history = reloaded.get_active_session().history
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0]["role"], "user")
        self.assertEqual(history[1]["role"], "model")


if __name__ == "__main__":
    unittest.main()
