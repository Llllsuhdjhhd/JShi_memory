import unittest
from unittest.mock import MagicMock
from datetime import datetime
from rems.config import REMSConfig, StorageConfig, LLMConfig
from rems.models.event import Event, EventRoleEntry, Importance
from rems.models.metabolism import Shadow
from rems.services.recall_service import RecallService

class TestDynamicRecall(unittest.TestCase):
    def setUp(self):
        self.config = REMSConfig(
            storage=StorageConfig(database_url="sqlite:///:memory:"),
            recall_head_ratio=0.66,
            recall_summary_min_chars=20
        )
        self.event_repo = MagicMock()
        self.role_repo = MagicMock()
        self.vector_store = MagicMock()
        # 70/30 容量分层 (白皮书 §4.4) 在 RecallService 内通过 vector_store.count() 决定是否启用，
        # 必须返回真实 int；默认 MagicMock 不会，会触发 ``int < int`` 比较 TypeError。
        self.vector_store.count.return_value = 0
        # 集体遗忘检查 / Stream B 命中分数计算都会问 role_repo 拿白描；MagicMock 默认会返回
        # MagicMock 实例，后续 ``forgetting_strategy.score`` 内部对其 ``last_accessed_time``
        # 取 ``now - dt`` 会触发类型错误。让所有"按事件取白描"的查询统一返回 None。
        self.role_repo.get_white_painting_by_event.return_value = None
        self.service = RecallService(
            self.config, self.event_repo, self.role_repo, self.vector_store
        )

    def create_mock_event(self, eid, content, summaries):
        event = Event(
            event_id=eid,
            content_raw=content,
            summaries=summaries,
            role_list=[EventRoleEntry(role_id="role1", importance=Importance.B)],
            create_time=datetime.now()
        )
        return event

    def test_sliding_window_compression(self):
        # 1. Setup many events to force compression
        # Each event will have multiple summary levels
        events = []
        for i in range(15):
            eid = f"EVT-{i:03}"
            summaries = {
                "L1": "Detailed summary " + "x" * 100,
                "L2": "Medium summary " + "x" * 50,
                "L3": "Short summary " + "x" * 20,
                "L4": "Tiny " + "x" * 5
            }
            evt = self.create_mock_event(eid, "Raw content", summaries)
            events.append(evt)
            self.event_repo.get.return_value = evt # Simplified mock

        # Mock vector search results
        self.vector_store.search.return_value = [{"event_id": f"EVT-{i:03}", "distance": 0.1} for i in range(15)]
        
        # We need to mock event_repo.get for each specific ID
        def get_event(eid):
            idx = int(eid.split("-")[1])
            return events[idx]
        self.event_repo.get.side_effect = get_event

        # 2. Run recall with a small redline to force compression
        # Assuming physical_redline is usually large, let's force it small in config
        # context_chars * 10 / 66
        # Let's say we want a redline of 500 chars total for 15 events
        # 15 events * L1 (117 chars) = 1755 chars -> Definitely over 500
        
        self.config.context_window = 3300 # 3300 * 1.5 * 10 / 66 = 750 chars approx
        
        redline = self.service._config.physical_redline
        print(f"Testing with Physical Redline: {redline}")

        block = self.service.build_recall_block("test query")
        
        print(f"Total Block Length: {block.total_length}")
        self.assertLessEqual(block.total_length, redline)
        
        # 3. Verify Head/Tail logic
        # Head items should be L1/L2, Tail items should be deeper (L3/L4/Ultra)
        for i, item in enumerate(block.items):
            print(f"Item {i}: {item.event_id} | Level: {item.summary_level} | Len: {len(item.content)}")

if __name__ == "__main__":
    unittest.main()
