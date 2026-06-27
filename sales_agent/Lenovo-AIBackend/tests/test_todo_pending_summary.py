"""Smoke tests for Sprint 2 US 1.3 Execute To-Do pending summary + list sort."""

from __future__ import annotations

from datetime import date, timedelta
from uuid import UUID, uuid4

from app.models.to_do_list import TblToDoList


def _seller() -> UUID:
    return uuid4()


def _insert_todo(
    db_session,
    *,
    seller_id: UUID,
    title: str,
    due_date,
    status: str = "Open",
):
    row = TblToDoList(
        task_title=title,
        type_tag="action",
        priority="high",
        status=status,
        due_date=due_date,
        seller_id=seller_id,
    )
    db_session.add(row)
    db_session.commit()
    db_session.refresh(row)
    return row


def test_todo_summary_requires_seller_id(client):
    response = client.get("/ai-api/todos/summary")
    assert response.status_code == 422
    assert "sellerId" in response.text


def test_todo_summary_counts_open_and_overdue(client, db_session):
    seller = _seller()
    today = date.today()
    _insert_todo(db_session, seller_id=seller, title="overdue", due_date=today - timedelta(days=2))
    _insert_todo(db_session, seller_id=seller, title="today", due_date=today)
    _insert_todo(db_session, seller_id=seller, title="future", due_date=today + timedelta(days=3))
    _insert_todo(
        db_session,
        seller_id=seller,
        title="done",
        due_date=today,
        status="Completed",
    )
    other = _seller()
    _insert_todo(db_session, seller_id=other, title="other seller", due_date=today)

    response = client.get("/ai-api/todos/summary", params={"sellerId": str(seller)})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["count"] == 3
    assert body["overdue_count"] == 1
    assert body["due_today_count"] == 1
    assert body["has_overdue"] is True
    assert body["badge_color"] == "red"
    assert body["label"] == "3 tasks pending"
    assert body["source"] == "ai"


def test_todo_list_sorts_overdue_before_today(client, db_session):
    seller = _seller()
    today = date.today()
    _insert_todo(db_session, seller_id=seller, title="future", due_date=today + timedelta(days=5))
    _insert_todo(db_session, seller_id=seller, title="today", due_date=today)
    _insert_todo(db_session, seller_id=seller, title="overdue", due_date=today - timedelta(days=1))

    response = client.get("/ai-api/todos", params={"sellerId": str(seller)})
    assert response.status_code == 200, response.text
    titles = [task["task_title"] for task in response.json()["tasks"]]
    assert titles == ["overdue", "today", "future"]


def test_todo_list_includes_overdue_tasks(client, db_session):
    seller = _seller()
    today = date.today()
    _insert_todo(db_session, seller_id=seller, title="late", due_date=today - timedelta(days=4))

    response = client.get("/ai-api/todos", params={"sellerId": str(seller)})
    assert response.status_code == 200, response.text
    assert response.json()["summary"]["overdue"] == 1
    assert len(response.json()["tasks"]) == 1


def test_todo_list_includes_linked_account_and_opportunity_names(client, db_session):
    from uuid import UUID

    from app.models.crm import CrmAccount, CrmOpportunity

    seller = _seller()
    account_id = UUID("a0000001-aaaa-0001-0001-000000000001")
    opportunity_id = UUID("b0000001-0001-0001-0001-000000000039")

    db_session.add(
        CrmAccount(
            accountid=str(account_id),
            name="JPMorgan Chase & Co.",
        )
    )
    db_session.add(
        CrmOpportunity(
            opportunityid=str(opportunity_id),
            name="JPMorgan – NY HQ Workstation Refresh",
            estimatedvalue=6750000,
        )
    )
    db_session.add(
        TblToDoList(
            task_title="Send security docs",
            type_tag="document",
            priority="High",
            status="Open",
            due_date=date.today(),
            seller_id=seller,
            linked_account_id=account_id,
            linked_opportunity_id=opportunity_id,
        )
    )
    db_session.commit()

    response = client.get("/ai-api/todos", params={"sellerId": str(seller)})
    assert response.status_code == 200, response.text
    task = response.json()["tasks"][0]
    assert task["account_name"] == "JPMorgan Chase & Co."
    assert task["opportunity_name"] == "JPMorgan – NY HQ Workstation Refresh"
    assert task["deal_value"] == 6750000.0
