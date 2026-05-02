import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.text import Text
import numpy as np

def parse_rules(rules_str):
    if len(rules_str) != 8 or not all(ch in '01?' for ch in rules_str):
        raise ValueError("Нужно 8 символов (0,1,?)")
    triplets = [
        (1,1,1), (1,1,0), (1,0,1), (1,0,0),
        (0,1,1), (0,1,0), (0,0,1), (0,0,0)
    ]
    rules = {}
    for i, ch in enumerate(rules_str):
        if ch == '?':
            rules[triplets[i]] = None
        else:
            rules[triplets[i]] = int(ch)
    return rules

def evolve(initial, rules, steps):
    generations = [initial[:]]
    for _ in range(steps):
        cur = generations[-1]
        n = len(cur)
        nxt = [0] * (n + 2)
        for i in range(n + 2):
            left  = cur[i-2] if 0 <= i-2 < n else 0
            mid   = cur[i-1] if 0 <= i-1 < n else 0
            right = cur[i]   if 0 <= i   < n else 0
            if left == 2 or mid == 2 or right == 2:
                nxt[i] = 2
                continue
            new = rules.get((left, mid, right))
            nxt[i] = 2 if new is None else new
        generations.append(nxt)
    return generations

def pad(generations):
    max_len = max(len(g) for g in generations)
    padded = []
    for g in generations:
        pad = max_len - len(g)
        left = pad // 2
        right = pad - left
        padded.append([0]*left + g + [0]*right)
    return padded

def ascii_display(matrix):
    for row in matrix:
        print(''.join('#' if x==1 else '.' if x==0 else '?' for x in row))

def draw_grid_with_questions(matrix, filename, cell_size=30):
  
    rows, cols = len(matrix), len(matrix[0])
    dpi = 100
    fig_width = cols * cell_size / dpi
    fig_height = rows * cell_size / dpi
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.invert_yaxis()  
    ax.set_aspect('equal')
    ax.axis('off')    

    for i in range(rows):
        for j in range(cols):
            val = matrix[i][j]
            if val == 1:
                facecolor = 'black'
                text = None
            elif val == 0:
                facecolor = 'white'
                text = None
            else:  
                facecolor = 'lightgray' 
                text = '?'
            rect = patches.Rectangle((j, i), 1, 1, linewidth=0.5,
                                     edgecolor='gray', facecolor=facecolor)
            ax.add_patch(rect)
            if text:
                ax.text(j + 0.5, i + 0.5, text,
                        ha='center', va='center', fontsize=cell_size*0.8,
                        fontweight='bold', color='black')

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(filename, dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    print(f"Изображение с сеткой сохранено в {filename}")

def main():
    print("=== Клеточный автомат c '?' на картинке ===\n")
    rule = input("Введите 8 символов (0,1,?): ").strip()
    try:
        rules = parse_rules(rule)
    except Exception as e:
        print("Ошибка:", e)
        return

    init = input("Начальная конфигурация (0/1, начинается и кончается 1): ").strip()
    if not all(c in '01' for c in init):
        print("Ошибка: только 0 и 1")
        return
    initial = [int(c) for c in init]

    steps = int(input("Количество шагов: "))
    print("Моделирование...")
    generations = evolve(initial, rules, steps)
    matrix = pad(generations)

    print("\nВыберите вывод:")
    print("1 – показать окно (не работает на GitHub)")
    print("2 – сохранить PNG с сеткой и '?'")
    print("3 – вывести в консоль (# . ?)")
    choice = input("1/2/3: ")

    if choice == '1':
        try:
            data = np.array(matrix)
            plt.imshow(data, cmap='gray_r', aspect='auto')
            plt.show()
        except:
            print("Не удалось показать. Сохраните в PNG.")
    elif choice == '2':
        fname = input("Имя PNG (например, out.png): ").strip() or "automaton.png"
        draw_grid_with_questions(matrix, fname, cell_size=30)
    elif choice == '3':
        ascii_display(matrix)
    else:
        print("Неверный выбор.")

if __name__ == "__main__":
    main()