from unittest.mock import patch

from pittqlab_utils.llm.pittai import PittAIClient, PittAIResponse
from pittqlab_utils.scoring import Rubric, RubricCategory, score_with_rubric


def _make_category(name, max_points=25):
    return RubricCategory(
        name=name,
        max_points=max_points,
        descriptor="Assess strength in this area.",
        levels={"high": "Excellent evidence", "mid": "Some evidence", "low": "Limited evidence"},
    )


@patch.object(PittAIClient, "chat")
def test_score_single_category_happy_path(mock_chat):
    mock_chat.return_value = PittAIResponse(
        text='{"score": 20, "justification": "Strong work.", "flags": []}',
        model="test-model",
    )
    rubric = Rubric(name="Admissions", categories=[_make_category("Research Experience", max_points=25)])
    client = PittAIClient(api_key="test-key")

    result = score_with_rubric("Candidate completed two internships.", rubric, client)

    assert result.category_scores[0].score == 20
    assert result.total_score == 20
    assert result.max_score == 25


@patch.object(PittAIClient, "chat")
def test_score_with_bonus_category(mock_chat):
    mock_chat.side_effect = [
        PittAIResponse(
            text='{"score": 20, "justification": "Strong baseline.", "flags": []}',
            model="test-model",
        ),
        PittAIResponse(
            text='{"score": 5, "justification": "Great bonus evidence.", "flags": []}',
            model="test-model",
        ),
    ]
    rubric = Rubric(
        name="Admissions",
        categories=[_make_category("Research Experience", max_points=25)],
        bonus_categories=[_make_category("Leadership Bonus", max_points=5)],
    )
    client = PittAIClient(api_key="test-key")

    result = score_with_rubric("Candidate has publications and leadership roles.", rubric, client)

    assert result.total_score == 25
    assert result.max_score == 25
    assert len(result.bonus_scores) == 1


@patch.object(PittAIClient, "chat")
def test_json_parse_failure_sets_low_confidence_flag(mock_chat):
    mock_chat.return_value = PittAIResponse(text="this is not json", model="test-model")
    rubric = Rubric(name="Admissions", categories=[_make_category("Research Experience", max_points=25)])
    client = PittAIClient(api_key="test-key")

    result = score_with_rubric("Candidate profile text.", rubric, client)

    assert result.category_scores[0].score == 0
    assert "low_confidence" in result.category_scores[0].flags


@patch.object(PittAIClient, "chat")
def test_api_error_does_not_abort_run(mock_chat):
    mock_chat.side_effect = [
        Exception("timeout"),
        PittAIResponse(
            text='{"score": 10, "justification": "Moderate evidence.", "flags": []}',
            model="test-model",
        ),
    ]
    rubric = Rubric(
        name="Admissions",
        categories=[
            _make_category("Research Experience", max_points=25),
            _make_category("Academic Fit", max_points=25),
        ],
    )
    client = PittAIClient(api_key="test-key")

    result = score_with_rubric("Candidate profile text.", rubric, client)

    assert len(result.category_scores) == 2
    assert "api_error" in result.category_scores[0].flags
    assert result.category_scores[1].score == 10


@patch.object(PittAIClient, "chat")
def test_flags_propagate_to_result(mock_chat):
    mock_chat.return_value = PittAIResponse(
        text='{"score": 15, "justification": "Some strength.", "flags": ["NV_experience"]}',
        model="test-model",
    )
    rubric = Rubric(name="Admissions", categories=[_make_category("Research Experience", max_points=25)])
    client = PittAIClient(api_key="test-key")

    result = score_with_rubric("Candidate profile text.", rubric, client)

    assert "NV_experience" in result.flags


@patch.object(PittAIClient, "chat")
def test_anchor_examples_appear_in_prompt(mock_chat):
    mock_chat.return_value = PittAIResponse(
        text='{"score": 19, "justification": "Well supported.", "flags": []}',
        model="test-model",
    )
    category = _make_category("Research Experience", max_points=25)
    rubric = Rubric(
        name="Admissions",
        categories=[category],
        anchor_examples=[
            {
                "text": "Student completed three projects.",
                "category": "Research Experience",
                "score": 22,
                "justification": "Shows sustained independent research output.",
            }
        ],
    )
    client = PittAIClient(api_key="test-key")

    score_with_rubric("Candidate profile text.", rubric, client)

    system_prompt = mock_chat.call_args.kwargs["system"]
    assert "Shows sustained independent research output." in system_prompt


@patch.object(PittAIClient, "chat")
def test_empty_document_returns_zero_scores(mock_chat):
    mock_chat.return_value = PittAIResponse(
        text='{"score": 0, "justification": "No content.", "flags": ["low_confidence"]}',
        model="test-model",
    )
    rubric = Rubric(
        name="Admissions",
        categories=[
            _make_category("Research Experience", max_points=25),
            _make_category("Academic Fit", max_points=25),
        ],
    )
    client = PittAIClient(api_key="test-key")

    result = score_with_rubric("", rubric, client)

    assert result.total_score == 0
