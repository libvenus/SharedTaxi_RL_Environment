"""Summary workflow.

Two-stage pipeline:

    user input (file path, optional)  ──►  read_text_file (tool)
                                                  │  (full file text)
                                                  ▼
                                            SummaryAgent  ──►  summary

If the user leaves the input blank in DevUI, the workflow falls back to
the bundled demo transcript at `<repo>/Demo.txt`. Otherwise it reads the
file at the given path. Standalone - not wired into the Sales / Analyst /
Orchestrator setup.
"""

from pathlib import Path

from agent_framework import WorkflowBuilder, WorkflowContext, executor

from summary_agent.agent import agent as summary_agent


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_TRANSCRIPT = REPO_ROOT / "Demo.txt"


@executor
async def read_text_file(path: str, ctx: WorkflowContext[str]) -> None:
    """Open the file at `path` and forward its full text to the summary agent.

    `path` is whatever string the user types into DevUI's workflow input box.
    Empty / whitespace-only input falls back to the bundled `Demo.txt`
    transcript so a one-click demo run just works.
    """
    target = Path(path.strip()) if path and path.strip() else DEFAULT_TRANSCRIPT
    text = target.read_text(encoding="utf-8")
    await ctx.send_message(text)


workflow = (
    WorkflowBuilder(
        name="SummaryWorkflow",
        description=(
            "Read a text file and have SummaryAgent summarise it. "
            f"Defaults to {DEFAULT_TRANSCRIPT.name} when no path is given."
        ),
        start_executor=read_text_file,
    )
    .add_edge(read_text_file, summary_agent)
    .build()
)
