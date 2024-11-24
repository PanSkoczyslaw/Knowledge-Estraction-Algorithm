import json

EXAMPLE_RECIPE = {'lp': {'facts': [],
                         'assumptions': [],
                         'clauses': [{'tag': 'Cl',
                                      'clHead': {'idx': 1, 'label': ''},
                                      'clPAtoms': [{'idx': 2, 'label': ''},
                                                   {'idx': 3, 'label': ''}],
                                      'clNAtoms': []},
                                     {'tag': 'Cl', 'clHead': {'idx': 2, 'label': ''},
                                      'clPAtoms': [{'idx': 3, 'label': ''}],
                                      'clNAtoms': []},
                                     {'tag': 'Cl', 'clHead': {'idx': 2, 'label': ''},
                                      'clPAtoms': [{'idx': 1, 'label': ''}],
                                      'clNAtoms': []}
                                     ]},
                  'abductive_goal': {'tag': 'Cl',
                                     'clHead': {'idx': 1, 'label': ''},
                                     'clPAtoms': [],
                                     'clNAtoms': []},
                  'factors': {'beta': 1.0,
                              'ahln': 1.0,
                              'r': 0.05,
                              'bias': 0.0,
                              'w': 0.1,
                              'amin': 0.1}}


class Tristate:

    def __init__(self, value=None):
        if any(value is v for v in (True, False, None)):
            self.value = value
        else:
            raise ValueError("Tristate value must be True, False, or None")

    def __eq__(self, other):
        return (self.value is other.value if isinstance(other, Tristate)
                else self.value is other)

    def __ne__(self, other):
        return not self == other

    def __bool__(self):
        raise TypeError("Tristate object may not be used as a Boolean")

    def __str__(self):
        return str(self.value)

    def __repr__(self):
        return "Tristate(%s)" % self.value

    def isFalse(self):
        return self.value is False

    def isTrue(self):
        return self.value is True

    def isNone(self):
        return self.value is None

    def __invert__(self):
        if self.isNone():
            return Tristate(None)
        return Tristate(not self.value)


def tristate_all(tristates: [Tristate]) -> Tristate:
    """
    Implementation of function all() for tristate logic (logic conjunction).

    :param tristates: list of tristates (list of Tristate)
    :return: Tristate

    """
    if all(map(lambda x: x.isTrue(), tristates)):
        return Tristate(True)
    if any(map(lambda x: x.isFalse(), tristates)):
        return Tristate(False)
    return Tristate(None)


def tristate_any(tristates: [Tristate]) -> Tristate:
    """
    Implementation of function any() for tristate logic ( logic alternative).

    :param tristates: list of tristates (list of Tristate)
    :return: Tristate

    """
    if any(map(lambda x: x.isTrue(), tristates)):
        return Tristate(True)
    if any(map(lambda x: x.isNone(), tristates)):
        return Tristate(None)
    return Tristate(False)


def tristate_implication(antecedent: Tristate, consequent: Tristate) -> Tristate:
    if antecedent.isFalse():
        return Tristate(True)
    if consequent.isTrue():
        return Tristate(True)
    if antecedent.isTrue() and consequent.isFalse():
        return Tristate(False)
    return Tristate(None)


class Factors:

    def __init__(self,
                 beta: float,
                 ahln: float,
                 r: float,
                 bias: float,
                 w: float,
                 amin: float):
        self.beta = beta
        self.ahln = ahln
        self.bias = bias
        self.amin = amin
        self.r = r
        self.w = w

    @staticmethod
    def from_dict(d: dict):
        return Factors(**d)

    def to_dict(self):
        return {'beta': self.beta,
                'ahln': self.ahln,
                'r': self.r,
                'bias': self.bias,
                'w': self.w,
                'amin': self.amin}


class Atom:

    def __init__(self, idx: int, label: str = 'n'):
        self.idx = idx
        self.label = label
        self.negated = 'n' in label

    def __eq__(self, other):
        if isinstance(other, Atom):
            return self.idx == other.idx
        raise TypeError(f"Atom can be compared only to another atom, not with {type(other)}")

    def __hash__(self):
        return self.idx

    def __repr__(self):
        return f"Atom idx: {self.idx}, label: {self.label}"

    def to_dict(self):
        return {"idx": self.idx, "label": self.label}

    def evaluate(self, positive, negative) -> Tristate:
        if self.negated:
            if self in negative:
                return Tristate(True)
            if self in positive:
                return Tristate(False)
            return Tristate(None)
        else:
            if self in positive:
                return Tristate(True)
            if self in positive:
                return Tristate(False)
            return Tristate(None)


def contradiction(positive: [Atom], negative: [Atom]) -> bool:
    return bool(set(positive).intersection(set(negative)))


class Clause:

    def __init__(self, head: Atom, positive: [Atom], negative: [Atom], tag: str = ''):
        self.head = head
        self.positive = positive
        self.negative = negative
        self.tag = tag

    @staticmethod
    def from_dict(d: dict):
        return Clause(head=Atom(**d['clHead']),
                      positive=[Atom(**spec) for spec in d['clPAtoms']],
                      negative=[Atom(**spec) for spec in d['clNAtoms']],
                      tag=d['tag'])

    def to_dict(self):
        return {"tag": self.tag,
                "clHead": self.head.to_dict(),
                "clPAtoms": [atom.to_dict() for atom in self.positive],
                "clNAtoms": [atom.to_dict() for atom in self.negative]}

    def calculate(self, positive: [Atom], negative: [Atom]) -> Tristate:
        assert not contradiction(positive, negative)

        antecedent_positive = [atom.evaluate(positive, negative) for atom in self.positive]
        antecedent_negative = [atom.evaluate(positive, negative) for atom in self.negative]

        return tristate_all(antecedent_positive + antecedent_negative)


class LogicProgram:

    def __init__(self, facts: [Clause], assumptions: [Clause], clauses: [Clause]):
        self.facts = facts
        self.assumptions = assumptions
        self.clauses = clauses
        self.all_clauses = self.facts + self.assumptions + self.clauses

    @staticmethod
    def from_dict(d: dict):
        return LogicProgram(facts=[Clause.from_dict(spec) for spec in d['facts']],
                            assumptions=[Clause.from_dict(spec) for spec in d['assumptions']],
                            clauses=[Clause.from_dict(spec) for spec in d['clauses']])

    @staticmethod
    def from_json(json_string: str):
        lp_dict = json.loads(json_string)
        if 'lp' in lp_dict:
            lp_dict = lp_dict['lp']
        return LogicProgram.from_dict(lp_dict)

    @staticmethod
    def from_file(fp: str):
        with open(fp, 'r') as file:
            json_string = file.read()
        return LogicProgram.from_json(json_string)

    def to_dict(self) -> dict:
        return {"facts": [cl.to_dict() for cl in self.facts],
                "assumptions": [cl.to_dict() for cl in self.assumptions],
                "clauses": [cl.to_dict() for cl in self.clauses]}

    def add_clause(self, clause: Clause):
        if not clause.positive or clause.negative:
            self.facts.append(clause)
        else:
            self.clauses.append(clause)
        self.all_clauses.append(clause)

    def to_json(self) -> str:
        return json.loads(self.to_dict())

    def tp_single_iteration(self, positive: [Atom], negative: [Atom]):
        new_positive = [clause.head for clause in self.all_clauses if clause.calculate(positive, negative).isTrue()]
        new_negative = [clause.head for clause in self.all_clauses if clause.calculate(positive, negative).isFalse()]
        print(new_positive, new_negative)
        return new_positive, new_negative

    def tp(self):
        new_positive, new_negative = [], []

        while True:
            positive, negative = new_positive, new_negative
            new_positive, new_negative = self.tp_single_iteration(positive, negative)
            if (set(positive) == set(new_positive)) and (set(negative) == set(new_negative)):
                break

        return new_positive, new_negative


