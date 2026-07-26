from services.execution.app.judge import RetryableJudgeError
from services.execution.app.main import _retryable_worker_exception


def test_execution_retry_classifier_distinguishes_infrastructure_failures() -> None:
    assert _retryable_worker_exception(RetryableJudgeError("provider unavailable"))
    assert _retryable_worker_exception(TimeoutError("artifact store timed out"))
    assert not _retryable_worker_exception(ValueError("invalid candidate action"))
