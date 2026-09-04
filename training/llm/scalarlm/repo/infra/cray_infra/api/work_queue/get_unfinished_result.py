from cray_infra.api.work_queue.get_in_memory_results import get_in_memory_results
from cray_infra.api.work_queue.get_group_request_id import get_group_request_id

async def get_unfinished_result(request_id):
    """
    Return the mutable per-request result entry, creating a placeholder
    if this is the first completion we've seen for this request_id.

    Returns None when the group isn't registered in this process's
    in-memory store — e.g. after a uvicorn auto-reload that wiped state
    while the generate worker still had finish_work POSTs to deliver,
    or after the group was already finalized and cleared by a peer
    finish_work call. Callers must handle None; without this guard a
    late POST would 500 on `None["results"]`, and the worker's retry
    logic would then flood the API with dead work.
    """
    group_request_id = get_group_request_id(request_id)
    in_memory_results = await get_in_memory_results(group_request_id)

    if in_memory_results is None:
        return None

    if not request_id in in_memory_results["results"]:
        in_memory_results["results"][request_id] = { "is_acked": False }

    return in_memory_results["results"][request_id]
