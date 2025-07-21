import time

from rich.console import Console
from rich.progress import BarColumn
from rich.progress import Progress
from rich.progress import SpinnerColumn
from rich.progress import TextColumn
from rich.progress import TimeElapsedColumn
from rich.progress import TimeRemainingColumn

console = Console()

with Progress(
    SpinnerColumn(),  # 左侧显示一个旋转器
    TextColumn("[progress.description]{task.description}"),  # 任务描述
    BarColumn(),  # 进度条本体
    TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),  # 百分比
    TimeRemainingColumn(),  # 预计剩余时间
    TimeElapsedColumn(),  # 已经过时间
    console=console,  # 指定控制台实例
    transient=True,  # 任务完成后自动清除进度条行
) as progress:
    # 添加第一个任务
    task1 = progress.add_task("[bold red]下载模型权重...[/bold red]", total=1000)
    # 添加第二个任务
    task2 = progress.add_task("[bold blue]训练 Epochs...[/bold blue]", total=500)
    # 添加第三个任务 (初始不可见)
    task3 = progress.add_task("[bold yellow]验证中...[/bold yellow]", total=200, visible=False)

    while not progress.finished:  # 循环直到所有任务完成
        if not progress.tasks[0].finished:
            progress.update(task1, advance=5)  # 模拟下载进度
        if not progress.tasks[1].finished:
            progress.update(task2, advance=3)  # 模拟训练进度

        if progress.tasks[1].completed >= 250 and not progress.tasks[2].visible:
            progress.tasks[2].visible = True  # 当训练达到一定进度时，显示验证任务

        if not progress.tasks[2].finished:
            progress.update(task3, advance=1)  # 模拟验证进度

        time.sleep(0.02)

console.print("[bold green]所有训练和验证任务已完成！[/bold green]")
