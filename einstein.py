import os

size = 5
ending_char = '0'

red = 0
green = 1
white = 2
blue = 3
yellow = 4

british = 5
swedish = 6
danish = 7
norwegian = 8
german = 9

tea = 10
coffee = 11
water = 12
beer = 13
milk = 14

prince = 15
blends = 16
pallmall = 17
bluemasters = 18
dunhill = 19

dog = 20
cat = 21
bird = 22
horse = 23
fish = 24

f = lambda a, b: a + size * b
color = f
nation = f
drink = f
cigar = f
pet = f

parameters = [
        (0,     4,      color),
        (5,     9,      nation),
        (10,    14,     drink),
        (15,    19,     cigar),
        (20,    24,     pet)]


def generate_formulae(start, end, category):
    formula = []
    for i in range(start, end + 1):
        # each house has a category
        houses = []
        for house in range(1, size + 1):
            houses.append(str(category(house, i)))
        houses.append(ending_char)
        formula.append(' '.join(houses))

        for h1 in range(1, size + 1):
            # each category only once
            for h2 in range(1, h1):
                formula.append('-{} -{} {}'.format(
                    category(h2, i), category(h1, i), ending_char
                ))
            # 1 category per house
            for j in range(start, end + 1):
                if j == i:
                    continue
                formula.append('-{} -{} {}'.format(
                    category(h1, i), category(h1, j), ending_char
                ))

    return os.linesep.join(formula)


def pair_relationship(cat1, prop1, cat2, prop2):
    formula = []
    for i in range(1, size + 1):
        formula.append('-{} {} {}'.format(
            cat1(i, prop1), cat2(i, prop2), ending_char
        ))
        formula.append('{} -{} {}'.format(
            cat1(i, prop1), cat2(i, prop2), ending_char
        ))
    return os.linesep.join(formula)


def neighbor(cat1, prop1, cat2, prop2):
    formula = [
        '-{} {} {}'.format(
            cat1(1, prop1), cat2(2, prop2), ending_char
        ),
        '-{} {} {}'.format(
            cat1(size, prop1), cat2(size - 1, prop2), ending_char
        )]
    for i in range(2, size):
        formula.append('-{} {} {} {}'.format(
            cat1(i, prop1), cat2(i-1, prop2), cat2(i+1, prop2), ending_char
        ))
    return os.linesep.join(formula)


def count_cnf(formula):
    literals = set()
    lines = formula.split(os.linesep)
    for line in lines:
        ls = line.split(' ')
        try:
            literals.update(list(map(lambda x: abs(int(x)), ls)))
        except ValueError:
            pass
    return 'p cnf {} {}'.format(len(literals)-1, len(lines))


def generate_einstein_formula():
    formula = []
    formula.extend([generate_formulae(*p) for p in parameters])

    # The Norwegian lives in the first house.
    formula.append('{} {}'.format(nation(1, norwegian), ending_char))
    # The Norwegian lives next to the blue house.
    formula.append('{} {}'.format(color(2, blue), ending_char))
    # The man living in the center house drinks milk.
    formula.append('{} {}'.format(drink(3, milk), ending_char))
    # The Brit lives in the red house.
    formula.append(pair_relationship(nation, british, color, red))
    # The green house’s owner drinks coffee.
    formula.append(pair_relationship(color, green, drink, coffee))
    # The Dane drinks tea.
    formula.append(pair_relationship(nation, danish, drink, tea))
    # The owner of the yellow house smokes Dunhill.
    formula.append(pair_relationship(color, yellow, cigar, dunhill))
    # The Swede keeps dogs as pets.
    formula.append(pair_relationship(nation, swedish, pet, dog))
    # The German smokes Prince.
    formula.append(pair_relationship(nation, german, cigar, prince))
    # The person who smokes Pall Mall rears birds.
    formula.append(pair_relationship(cigar, pallmall, pet, bird))
    # The owner who smokes Bluemasters drinks beer.
    formula.append(pair_relationship(cigar, bluemasters, drink, beer))
    # The man who keeps the horse lives next to the man who smokes Dunhill.
    formula.append(neighbor(pet, horse, cigar, dunhill))
    # The man who smokes Blends lives next to the one who keeps cats.
    formula.append(neighbor(cigar, blends, pet, cat))
    # The man who smokes Blends has a neighbor who drinks water.
    formula.append(neighbor(cigar, blends, drink, water))
    # The green house is on the left of the white house.
    for w in range(1, size+1):
        for g in range(size, 0, -1):
            if w-1 <= g <= w:
                continue
            formula.append('-{} -{} {}'.format(
                color(w, white), color(g, green), ending_char
            ))
    formula_strings = os.linesep.join(formula)
    formula_cnf = os.linesep.join([count_cnf(formula_strings), formula_strings])
    formula_final = os.linesep.join(["c Einstein's puzzle, encoded in CNF", formula_cnf])
    return formula_final

def generate_legend():
    refs = []
    for i in range(1, size+1):
        for c in ['red', 'green', 'white', 'blue', 'yellow']:
            refs.append('{:4.0f}: color({}, {})'.format(color(i, eval(c)), i, c))
        for n in ['british', 'swedish', 'danish', 'norwegian', 'german']:
            refs.append('{:4.0f}: nation({}, {})'.format(nation(i, eval(n)), i, n))
        for d in ['tea', 'coffee', 'water', 'beer', 'milk']:
            refs.append('{:4.0f}: drink({}, {})'.format(drink(i, eval(d)), i, d))
        for c in ['prince', 'blends', 'pallmall', 'bluemasters', 'dunhill']:
            refs.append('{:4.0f}: cigar({}, {})'.format(cigar(i, eval(c)), i, c))
        for p in ['dog', 'cat', 'bird', 'horse', 'fish']:
            refs.append('{:4.0f}: pet({}, {})'.format(pet(i, eval(p)), i, p))
    return os.linesep.join(sorted(refs))


if __name__ == '__main__':
    with open('einstein.txt', 'w') as f:
        formula = generate_einstein_formula()
        f.write(formula)
    with open('legend.txt', 'w') as f:
        f.write(generate_legend())