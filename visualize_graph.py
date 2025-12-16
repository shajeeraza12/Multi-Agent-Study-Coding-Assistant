# visualize_graph.py
import os
from graph import app


def save_graph_image():
    """
    Saves a Mermaid diagram of the LangGraph structure to the 'assets' folder.
    """
    try:
        os.makedirs("assets", exist_ok=True)

        # Get the mermaid diagram as text (no external API, no browser)
        graph_repr = app.get_graph().draw_mermaid()

        output_path = os.path.join("assets", "research_graph.mmd")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(graph_repr)

        print(f"✓ Graph saved as Mermaid diagram to {output_path}")
        print("You can view or export it to PNG at: https://mermaid.live/")
    except Exception as e:
        print(f"An error occurred while generating the graph: {e}")


if __name__ == "__main__":
    save_graph_image()
