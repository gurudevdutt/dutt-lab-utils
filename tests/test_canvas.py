import importlib
import inspect
import json
import sys
from unittest.mock import MagicMock, mock_open, patch

import pytest


def _load_canvas_module():
    module_name = "pittqlab_utils.canvas"
    sys.modules.pop(module_name, None)
    with patch("builtins.open", mock_open(read_data=json.dumps({"token": "token-123", "course_id": "42"}))):
        return importlib.import_module(module_name)


@pytest.fixture
def canvas_module():
    module = _load_canvas_module()
    yield module
    sys.modules.pop("pittqlab_utils.canvas", None)


def _make_response(json_data, links=None, headers=None, status_code=200, content=b""):
    response = MagicMock()
    response.json.return_value = json_data
    response.links = links or {}
    response.headers = headers or {}
    response.status_code = status_code
    response.content = content
    response.raise_for_status.return_value = None
    return response


def test_get_happy_path_and_pagination(canvas_module):
    first = _make_response([{"id": 1}], links={"next": {"url": "https://next"}})
    second = _make_response([{"id": 2}], links={})
    with patch.object(canvas_module.requests, "get", side_effect=[first, second]) as mock_get:
        result = canvas_module.get("assignments")
    assert result == [{"id": 1}, {"id": 2}]
    mock_get.assert_any_call(
        "https://canvas.pitt.edu/api/v1/courses/42/assignments",
        headers={"Authorization": "Bearer token-123"},
        json=None,
    )
    mock_get.assert_any_call("https://next", headers={"Authorization": "Bearer token-123"})


def test_post_happy_path(canvas_module):
    response = _make_response({"ok": True})
    with patch.object(canvas_module.requests, "post", return_value=response) as mock_post:
        canvas_module.post("groups", {"name": "g1"})
    mock_post.assert_called_once_with(
        "https://canvas.pitt.edu/api/v1/courses/42/groups",
        headers={"Authorization": "Bearer token-123"},
        json={"name": "g1"},
    )


def test_put_happy_path(canvas_module):
    response = _make_response({"ok": True})
    with patch.object(canvas_module.requests, "put", return_value=response) as mock_put:
        canvas_module.put("groups/1", {"name": "g2"})
    mock_put.assert_called_once_with(
        "https://canvas.pitt.edu/api/v1/courses/42/groups/1",
        headers={"Authorization": "Bearer token-123"},
        json={"name": "g2"},
    )


def test_upload_happy_path(canvas_module):
    step1 = _make_response({"upload_url": "https://upload-url", "upload_params": {"k": "v"}})
    step2 = _make_response({}, headers={"Location": "https://location-url"})
    step3 = _make_response({"id": 99})
    with patch.object(canvas_module.os.path, "basename", return_value="sample.pdf"), patch.object(
        canvas_module.os.path, "getsize", return_value=123
    ), patch.object(canvas_module.requests, "post", side_effect=[step1, step2]), patch.object(
        canvas_module.requests, "get", return_value=step3
    ), patch.object(
        canvas_module, "open", mock_open(read_data=b"file-bytes")
    ):
        result = canvas_module.upload("files", "/tmp/sample.pdf")
    assert result == {"id": 99}


def test_download_attachment_happy_path(canvas_module):
    response = _make_response({}, content=b"payload")
    with patch.object(canvas_module.os.path, "exists", return_value=False), patch.object(
        canvas_module.requests, "get", return_value=response
    ) as mock_get, patch.object(canvas_module, "open", mock_open()) as mock_file:
        output = canvas_module.download_attachment("https://file-url", "/tmp/file.bin")
    assert output == "/tmp/file.bin"
    mock_get.assert_called_once_with("https://file-url", headers={"Authorization": "Bearer token-123"})
    mock_file.assert_called_once_with("/tmp/file.bin", "wb")


def test_download_attachment_is_idempotent(canvas_module):
    with patch.object(canvas_module.os.path, "exists", return_value=True), patch.object(
        canvas_module.requests, "get"
    ) as mock_get, patch.object(canvas_module, "open", mock_open()) as mock_file:
        output = canvas_module.download_attachment("https://file-url", "/tmp/file.bin")
    assert output == "/tmp/file.bin"
    mock_get.assert_not_called()
    mock_file.assert_not_called()


@pytest.mark.parametrize(
    ("func_name", "args", "expected_path", "expected"),
    [
        ("get_groups", (), "groups", [{"id": "g"}]),
        ("get_students", (), "students", [{"id": "s"}]),
        ("get_assignments", (), "assignments", [{"id": "a"}]),
        ("get_assignment", (11,), "assignments/11", {"id": 11}),
        ("get_submissions", (7,), "assignments/7/submissions", [{"user_id": 1}]),
        ("get_enrollments", (), "enrollments", [{"id": "e"}]),
    ],
)
def test_helper_get_wrappers_happy_path(canvas_module, func_name, args, expected_path, expected):
    with patch.object(canvas_module, "get", return_value=expected) as mock_get:
        result = getattr(canvas_module, func_name)(*args)
    assert result == expected
    mock_get.assert_called_once_with(expected_path)


def test_get_submissions_skip_unsubmitted_if_supported(canvas_module):
    submissions = [
        {"id": 1, "workflow_state": "submitted"},
        {"id": 2, "workflow_state": "unsubmitted"},
    ]
    sig = inspect.signature(canvas_module.get_submissions)
    if "skip_unsubmitted" in sig.parameters:
        with patch.object(canvas_module, "get", return_value=submissions):
            result = canvas_module.get_submissions(99, skip_unsubmitted=True)
        assert result == [{"id": 1, "workflow_state": "submitted"}]
    else:
        with patch.object(canvas_module, "get", return_value=submissions):
            result = canvas_module.get_submissions(99)
        assert result == submissions


def test_get_group_users_happy_path_with_pagination(canvas_module):
    first = _make_response([{"id": 1}], links={"next": {"url": "https://next-users"}})
    second = _make_response([{"id": 2}], links={})
    with patch.object(canvas_module.requests, "get", side_effect=[first, second]) as mock_get:
        result = canvas_module.get_group_users(123)
    assert result == [{"id": 1}, {"id": 2}]
    mock_get.assert_any_call(
        "https://canvas.pitt.edu/api/v1/groups/123/users",
        headers={"Authorization": "Bearer token-123"},
    )
    mock_get.assert_any_call("https://next-users", headers={"Authorization": "Bearer token-123"})


def test_assign_peer_review_happy_path(canvas_module):
    response = _make_response({"id": 555, "user_id": 33})
    with patch.object(canvas_module.requests, "post", return_value=response) as mock_post:
        result = canvas_module.assign_peer_review(10, 20, 33)
    assert result == {"id": 555, "user_id": 33}
    mock_post.assert_called_once_with(
        "https://canvas.pitt.edu/api/v1/courses/42/assignments/10/submissions/20/peer_reviews",
        headers={"Authorization": "Bearer token-123"},
        json={"user_id": 33},
    )


def test_get_attendance_happy_path(canvas_module):
    with patch.object(canvas_module, "get", return_value=[{"present": True}]) as mock_get:
        result = canvas_module.get_attendance()
    assert result == [{"present": True}]
    mock_get.assert_called_once_with("attendance")


def test_get_student_activity_happy_path_specific_user(canvas_module):
    with patch.object(canvas_module, "get_assignments", return_value=[{"id": 1, "name": "HW1"}]), patch.object(
        canvas_module,
        "get_submissions",
        return_value=[{"user_id": 5, "submitted_at": "2026-01-01", "workflow_state": "submitted"}],
    ):
        result = canvas_module.get_student_activity(user_id=5, limit_assignments=1)
    assert result == {"HW1": {"submitted": "2026-01-01", "status": "submitted"}}


def test_get_student_activity_happy_path_all_students(canvas_module):
    students = [{"id": 1, "name": "Alice"}]
    assignments = [{"id": 11, "name": "A11"}]
    submissions = [{"user_id": 1, "submitted_at": "2026-01-03", "workflow_state": "submitted"}]
    with patch.object(canvas_module, "get_students", return_value=students), patch.object(
        canvas_module, "get_assignments", return_value=assignments
    ), patch.object(canvas_module, "get_submissions", return_value=submissions):
        result = canvas_module.get_student_activity(user_id=None, limit_assignments=1)
    assert result["Alice"]["total_assignments_checked"] == 1
    assert result["Alice"]["submissions"] == 1
    assert result["Alice"]["last_activity"] == "2026-01-03"


def test_get_roll_call_attendance_happy_path(canvas_module):
    tools_resp = _make_response([{"id": 8, "name": "Roll Call Attendance"}])
    endpoint_resp = _make_response({"sessions": []}, status_code=200)
    with patch.object(canvas_module.requests, "get", side_effect=[tools_resp, endpoint_resp]):
        result = canvas_module.get_roll_call_attendance()
    assert result == {"sessions": []}


def test_canvas_client_from_env_happy_path():
    module = _load_canvas_module()
    with patch.dict("os.environ", {"CANVAS_TOKEN": "env-token", "CANVAS_COURSE_ID": "999"}, clear=True):
        result = module.canvas_client_from_env()
    assert result == {"token": "env-token", "course_id": "999"}


def test_canvas_client_from_env_raises_on_missing_env_vars():
    module = _load_canvas_module()
    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(ValueError, match="CANVAS_TOKEN"):
            module.canvas_client_from_env()
