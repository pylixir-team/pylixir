from pylixir.core.progress import Progress


def show_progress(progress: Progress) -> str:
    return f"Turn left: {progress.get_turn_left()} | reroll left: {progress.get_reroll_left()}"
