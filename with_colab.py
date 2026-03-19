from IPython.display import display, HTML

with open('tictactoe.html', 'r', encoding='utf-8') as f:
    display(HTML(f.read()))
