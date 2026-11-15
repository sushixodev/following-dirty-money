from functools import partial

import gradio as gr

import core

EXAMPLES = {"illicit": "Random illicit case", "licit": "Random licit case", "error": "Random model error",
            "random": "Random transaction"}

HEAD = """<script>
document.addEventListener("wheel", event => {
  if (!event.ctrlKey && event.target.closest(".js-plotly-plot")) event.stopPropagation();
}, {capture: true, passive: true});
</script>"""


def inspect(text):
    explanation, scores, chart, linked = core.inspect(text)
    return explanation, scores, chart, gr.Dropdown(choices=linked, value=None)


def open_row(ids, event: gr.SelectData):
    return ids[event.index[0]], gr.Tabs(selected="check")


with gr.Blocks(title="Following Dirty Money") as demo:
    gr.Markdown("# Following Dirty Money\n"
                "Scores Bitcoin transactions from the Elliptic dataset by how likely they are to be money laundering. "
                "The random examples come from periods the model never saw during training. Veritas AI Hackathon "
                "2026, AI + Finance track. Prototype, not financial advice.")

    with gr.Tabs() as tabs:
        with gr.Tab("Check a transaction", id="check"):
            with gr.Row():
                box = gr.Textbox(label="Transaction ID", scale=4)
                check_button = gr.Button("Check", variant="primary", scale=1)
            with gr.Row():
                example_buttons = {kind: gr.Button(label, size="sm") for kind, label in EXAMPLES.items()}
            with gr.Row():
                explanation = gr.Markdown()
                scores = gr.Markdown()
            gr.Markdown("**Neighbourhood in the payment graph.** Each point is a transaction and each line a payment, "
                        "up to two steps away. Size shows the number of links and colour the risk; the selected "
                        "transaction has a black outline. Drag to rotate, Ctrl + scroll to zoom.")
            chart = gr.Plot(show_label=False)
            explore = gr.Dropdown(label="Open a linked transaction (highest risk first)", choices=[])

        with gr.Tab("Review queue", id="queue"):
            gr.Markdown("An analyst can review only a limited number of transactions per period. The table lists "
                        "those with the highest risk. Click a row to open it.")
            with gr.Row():
                period = gr.Slider(core.PERIODS[0], core.PERIODS[-1], value=40, step=1, label="Period")
                budget = gr.Slider(5, 50, value=20, step=5, label="Number of reviews")
            reveal = gr.Checkbox(label="Show ground truth")
            summary = gr.Markdown()
            table = gr.Dataframe(interactive=False)
            queue_ids = gr.State([])

        with gr.Tab("Model health", id="health"):
            gr.Markdown("Labels arrive months late, so the model is watched with three signals each period. Move the "
                        "slider to replay the test periods.")
            health_period = gr.Slider(core.PERIODS[0], core.PERIODS[-1], value=core.BREAK - 1, step=1, label="Period")
            health_table = gr.Markdown()
            health_chart = gr.Plot(show_label=False)

        with gr.Tab("About", id="about"):
            gr.Markdown(core.about())

    shown = [explanation, scores, chart, explore]
    check_button.click(inspect, box, shown)
    box.submit(inspect, box, shown)
    for kind, button in example_buttons.items():
        button.click(partial(core.example, kind), None, box).then(inspect, box, shown)
    explore.input(lambda tx: tx, explore, box).then(inspect, box, shown)

    queue_inputs, queue_outputs = [period, budget, reveal], [summary, table, queue_ids]
    for control in queue_inputs:
        control.change(core.queue, queue_inputs, queue_outputs)
    table.select(open_row, queue_ids, [box, tabs]).then(inspect, box, shown)

    health_period.change(core.health_view, health_period, [health_table, health_chart])

    demo.load(partial(core.example, "illicit"), None, box).then(inspect, box, shown)
    demo.load(core.queue, queue_inputs, queue_outputs)
    demo.load(core.health_view, health_period, [health_table, health_chart])

if __name__ == "__main__":
    demo.launch(head=HEAD)
