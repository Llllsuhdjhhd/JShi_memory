from unittest.mock import Mock
from sqlalchemy import event as sql_event
from rems.models.event import Event, EventRoleEntry
from rems.models.role import WhitePaintingEntry
from rems.recall import RecallPipeline, HashEmbedding
from rems.storage.repository import EventRepository, RoleRepository


def test_lexical_match_normalizes_only_before_matching():
    from rems.recall.pipeline import lexical_score
    seen = []
    def normalize(text):
        seen.append(text)
        return text.replace('車', '车')
    assert lexical_score('修車计划', '今天修车', normalizer=normalize) > 0
    assert seen == ['修車计划', '今天修车']


def test_bulk_read_chunks_missing_and_duplicates(db):
    repo = RoleRepository(db)
    for text in ['first', 'second']:
        repo.add_white_painting_entry('p', WhitePaintingEntry(event_id='e', role_summary=text))
    queries = []
    sql_event.listen(db.engine, 'before_cursor_execute', lambda *args: queries.append(args[2]))
    result = repo.white_paintings_for_events(['e', 'e', *map(str, range(1001))])
    assert len(queries) == 3
    assert result[('p', 'e')].role_summary == 'first'
    assert ('p', 'missing') not in result
    queries.clear()
    assert repo.white_paintings_for_events([]) == {}
    assert not queries


def test_fact_snapshot_no_extra_reads_and_refinement_priority(db):
    events, roles = EventRepository(db), RoleRepository(db)
    pipe = RecallPipeline(HashEmbedding(), None, events, role_repo=roles)
    ev = Event(event_id='e', subject_id='s', content_raw='hello',
               role_list=[EventRoleEntry(role_id='p')], refinement_object_levels={'p': {'L1': 'new'}})
    events.get = Mock(side_effect=AssertionError('event already loaded'))
    roles.get_white_painting_by_event = Mock(side_effect=AssertionError('facts already loaded'))
    entries = {('p', 'e'): WhitePaintingEntry(event_id='e', role_summary='old')}
    assert pipe._fact_levels('p', 'e', event=ev, entries=entries) == {'L1': 'new'}
    assert pipe._fact_levels('absent', 'e', event=ev, entries=entries) == {}
    assert pipe._fact_levels('p', 'e', event=ev.model_copy(update={'refinement_object_levels': {}}),
                             entries=entries) == {'L1': 'old'}


def test_recall_reads_fresh_facts_each_time(db):
    events, roles = EventRepository(db), RoleRepository(db)
    ev = Event(subject_id='s', content_raw='bicycle', summaries={'L1': 'bicycle'},
               role_list=[EventRoleEntry(role_id='p')])
    events.save(ev)
    pipe = RecallPipeline(HashEmbedding(), None, events, role_repo=roles)
    original = roles.white_paintings_for_events
    roles.white_paintings_for_events = Mock(wraps=original)
    pipe.recall('s', 'bicycle', channels=('lexical',), reinforce=False)
    roles.add_white_painting_entry('p', WhitePaintingEntry(event_id=ev.event_id, role_summary='bicycle repair'))
    pipe.recall('s', 'bicycle', channels=('lexical',), reinforce=False)
    assert roles.white_paintings_for_events.call_count == 2
    assert original([ev.event_id])[('p', ev.event_id)].role_summary == 'bicycle repair'
