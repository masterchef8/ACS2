# coding=utf-8
"""
@author : somebody
@date : 04/06/16

@project : ACS2
@file : ACS2

@Class description :
"""
import copy

from Interface import Maze
from Classifier import Classifier
from Constant import Constants
from random import random, choice
import matplotlib.pyplot as plt

cons = Constants()


def does_match(cl, percept):
    """

    :param cl:
    :param percept:
    :return:
    """
    for i in range(len(percept)):
        if cl.condition[i] != cons.symbol and cl.condition[i] != percept[i]:
            return False
    return True


def gen_match_set(pop: list, percept: list):
    """

    :param pop:
    :type pop: list
    :param percept:
    :type percept: list
    :return:
    :rtype: list
    """
    m = []
    if time == 0:
        for i in range(cons.nbAction):
            newcl = Classifier()
            newcl.condition = [cons.symbol] * cons.lenCondition
            newcl.action = i
            newcl.effect = [cons.symbol] * cons.lenCondition
            newcl.exp = 0
            newcl.t = time
            newcl.q = 0.5
            pop.append(newcl)
    for cl in pop:
        if does_match(cl, percept):
            m.append(cl)
    return m


def update_application_average(cl: Classifier, t: int):
    """

    :type t: int
    :param t: Time
    :type cl: Classifier
    """
    if cl.exp < 1 / cons.beta:
        cl.aav += (t - cl.tga - cl.aav) / cl.exp
    else:
        cl.aav += cons.beta * (t - cl.tga - cl.aav)
    cl.tga = t


def does_anticipate_correctly(cl: Classifier, percept: list, percept_: list) -> bool:
    for i in range(cons.lenCondition):
        if cl.effect != cons.symbol:
            if percept_[i] != percept[i]:
                return False
        else:
            if cl.effect[i] != percept[i] or percept_[i] == percept[i]:
                return False
    return True


def get_differences(mark, percept: list) -> list:
    """

    :param mark:
    :param percept:
    :return: Differences between env and cl.mark
    """
    diff = [cons.symbol] * cons.lenCondition
    if mark is not None:
        type1 = 0
        type2 = 0
        for i in range(len(percept)):
            if mark[i] != percept[i]:
                type1 += 1
                # type2 += 1

        if type1 > 0:
            type1 = random() * type1
            for i in range(len(percept)):
                if mark[i] != percept[i]:
                    if int(type1) == 0:
                        diff[i] = percept[i]
                    type1 -= 1
        elif type2 > 0:
            for i in range(len(percept)):
                if mark[i] != percept[i]:
                    diff[i] = percept[i]
    return diff


def number_of_spec(condition: list) -> int:
    """

    :param condition:
    :return: Number of non-# in cl.condition or diff
    """
    n = 0
    for i in range(len(condition)):
        if condition[i] != cons.symbol:
            n += 1
    return n


def remove_random_spec_att(l: list):
    """

    :param l: list to modify
    """
    t = True
    while t:
        for i in l:
            if i != cons.symbol and random() > cons.specAtt:
                i = cons.symbol
                t = False


def expected_case(cl: Classifier, percept: list) -> Classifier:
    """

    :rtype: Classifier
    """
    diff = get_differences(cl.mark, percept)
    if diff == [cons.symbol] * cons.lenCondition:
        cl.q += cons.beta * (1 - cl.q)
        return None
    else:
        spec = number_of_spec(cl.condition)
        spec_new = number_of_spec(diff)
        child = copy.copy(cl)
        if spec == cons.uMax:
            remove_random_spec_att(child.condition)
            spec -= 1
            while spec + spec_new > cons.beta:
                if spec > 0 and random < 0.5:
                    remove_random_spec_att(child.condition)
                    spec -= 1
                else:
                    remove_random_spec_att(diff)
                    spec_new -= 1
        else:
            while spec + spec_new > cons.beta:
                remove_random_spec_att(diff)
                spec_new -= 1
        child.condition = diff
        if child.q < 0.5:
            child.q = 0.5
        child.exp = 1
        assert isinstance(child, Classifier), 'Should be a Classifier'
        return child


def unexpected_case(cl: Classifier, percept: list, percept_: list) -> Classifier:
    """

    :rtype: Classifier
    """
    cl.q = cl.q - cons.beta * cl.q
    cl.mark = percept_
    for i in range(len(percept)):
        if cl.effect[i] != cons.symbol:
            if cl.effect[i] != percept_[i] or percept_[i] != percept[i]:
                return None
    child = copy.copy(cl)
    for i in range(len(percept)):
        if cl.effect[i] == cons.symbol and percept_[i] != percept[i]:
            child.condition[i] = percept_[i]
            child.effect = percept[i]
    if cl.q < 0.5:
        cl.q = 0.5
    child.exp = 1
    return child


def add_alp_classifier(cl: Classifier, pop: list, aset: list):
    oldcl = None
    for c in aset:
        if c.isSubsumer(cl):
            if oldcl is None or c.isMoreGeneral(cl):
                oldcl = c
    if oldcl is None:
        for c in aset:
            if c.equals(cl):
                oldcl = c
    if oldcl is None:
        pop.append(cl)
        aset.append(cl)
    else:
        oldcl.q += cons.beta * (1 - oldcl.q)


def cover_triple(percept_: list, act: int, percept: list, t: int) -> Classifier:
    child = Classifier()

    for i in range(len(percept)):
        if percept_[i] != percept[i]:
            child.condition[i] = percept_[i]
            child.effect[i] = percept[i]
    child.action = act
    child.exp = 0
    child.r = 0
    child.aav = 0
    child.alp = t
    child.tga = t
    child.t = t
    child.q = 0.5
    child.num = 1
    return child


def apply_alp(aset: list, percept_: list, percept: list, t: int, pop: list):
    """

    :param aset:
    :type aset: list
    :param percept_:
    :type percept_: list
    :param percept:
    :type percept: list
    :param t:
    :type t: int
    :param pop:
    :type pop: list
    """
    wasExpectedCase = 0
    for cl in aset:
        cl.exp += 1
        update_application_average(cl, t)

        if does_anticipate_correctly(cl, percept, percept_):
            newcl = expected_case(cl, percept)
            wasExpectedCase += 1
        else:
            newcl = unexpected_case(cl, percept, percept_)
            if cl.q < cons.thetaI:
                aset.remove(cl)
                pop.remove(cl)

        if newcl is not None:
            newcl.tga = t
            add_alp_classifier(newcl, pop, aset)
    if wasExpectedCase == 0:
        newcl = cover_triple(percept_, act, percept, t)
        add_alp_classifier(newcl, pop, aset)


def get_max_for_rl(m: list) -> int:
    """

    :param m:
    :return:
    """
    sumrl = 0
    for cl in m:
        if cl.effect != [cons.symbol] * cons.lenCondition:
            sumrl += cl.q * cl.r
    return int(sumrl)


def apply_rl(aset: list, p: int):
    """


    :param aset:
    :param p:
    :param maxp:
    :type maxp: int
    """
    maxp = get_max_for_rl(M)
    for cl in aset:
        cl.r += cons.beta * (p + cons.gamma * maxp - cl.r)
        cl.ir += cons.beta * (p - cl.ir)


def select_offspring(aset: list) -> Classifier:
    qualitySum = 0
    for cl in aset:
        qualitySum += cl.q ** 3
    choicePoint = random() * qualitySum
    qualitySum = 0
    for cl in aset:
        qualitySum += cl.q ** 3
        if qualitySum > choicePoint:
            return cl


def apply_ga_mutation(cl: Classifier):
    for i in range(len(cl.condition)):
        if cl.condition[i] != cons.symbol:
            if random() < cons.mu:
                cl.condition[i] = cons.symbol


def apply_crossover(cl1: Classifier, cl2: Classifier):
    if cl1.effect != cl2.effect:
        return
    x = random()*(len(cl1.condition)+1)
    while True:
        y = random()*(len(cl1.condition)+1)
        if x != y:
            break

    if x > y:
        tmp = x
        x = y
        y = tmp
    i = 0
    while True:
        if x <= i < y:
            tp = cl1.condition[i]
            cl1.condition[i] = cl2.condition[i]
            cl2.condition[i] = tp
        i+=1
        if i <= y:
            break

def delete_classifier(aset: list, pop: list):
    sum = 0
    for c in pop:
        sum += c.num
    while cons.inSize + sum > cons.thetaAS:
        cldel = None
        for cl in pop:
            if random() < 1/3:
                if cldel is None:
                    cldel = cl
                else:
                    if cl.q - cldel.q < -0.1:
                        cldel = cl
                    if abs(cl.q - cldel.q) <= 0.1:
                        if cl.mark is not None and cldel.mark is None:
                            cldel = cl
                        elif cl.mark is not None or cldel.mark is None:
                            if cl.aav > cldel.aav:
                                cldel = cl

        if cldel is not None:
            if cldel.num > 1:
                cldel.num -= 1
            else:
                aset.remove(cl)
                pop.remove(cl)



def add_ga_classifier(aset: list, pop: list, cl: Classifier):
    oldcl = None
    for c in aset:
        if c.isSubsumer(cl):
            if oldcl is None or c.isMoreGeneral(cl):
                oldcl = c
    if oldcl is None:
        for c in aset:
            if c.equals(cl):
                oldcl = c
    if oldcl is None:
        pop.append(cl)
        aset.append(cl)
    else:
        if oldcl.mark is None:
            oldcl.num += 1


def apply_ga(aset: list, t: int, pop: list):
    sumNum = 0
    sumTgaN = 0
    for cl in aset:
        sumTgaN += cl.tga * cl.num
        sumNum += cl.num
    if t - sumTgaN / sumNum > cons.thetaGA:
        for cl in aset:
            cl.tga = t
        parent1 = select_offspring(aset)
        parent2 = select_offspring(aset)

        child1 = copy.copy(parent1)
        child2 = copy.copy(parent2)

        child1.num += 1
        child2.num += 1

        child1.exp += 1
        child2.exp += 1

        apply_ga_mutation(child1)
        apply_ga_mutation(child2)

        if random() < cons.x:
            apply_crossover(child1, child2)
            child1.r = (parent1.r + parent2.r) / 2
            child2.r = (parent1.r + parent2.r) / 2

            child1.q = (parent1.q + parent2.q) / 2
            child2.q = (parent1.q + parent2.q) / 2
        child1.q /= 2
        child2.q /= 2

        delete_classifier(aset, pop)
        if child1.condition == [cons.symbol]*cons.lenCondition:
            pass
        else:
            add_ga_classifier(aset, pop, child1)

        if child2.condition == [cons.symbol] * cons.lenCondition:
            pass
        else:
            add_ga_classifier(aset, pop, child2)


def choose_action(aset: list):
    if random() < cons.epsilon:
        c = choice([i for i in range(cons.nbAction)])
        return c
    else:
        bestCL = Classifier()
        for cl in aset:
            if cl.effect != [cons.symbol] * cons.lenCondition:
                bestCL = cl
                break
        for cl in aset:
            if cl.effect != [cons.symbol] * cons.lenCondition and cl.q * cl.r > bestCL.q * bestCL.r:
                bestCL = cl
        return bestCL.action


def gen_action_set(aset: list, act: int):
    """

    :param act: action
    :type act: int
    :param aset: Match set
    :type aset: list
    :rtype: list
    """
    a = []
    for cl in aset:
        if cl.action == act:
            a.append(cl)
    return a


if __name__ == '__main__':
    env = Maze()
    perf = []
    perfFitness = []
    perfClasseur = []
    meanQ = []
    boucle1 = True
    Pop = []
    M = []
    A = []
    A_ = None
    perception = None
    perception_ = None
    time = 0
    p = 0
    act = None
    m=0
    while True:
        perception = env.perception

        M = gen_match_set(Pop, perception)
        if len(A) == 0:
            pass
        if A_ is not None:
            apply_alp(A_, perception_, perception, time, Pop)
            apply_rl(A_, p)
            apply_ga(A_, time, Pop)

        act = choose_action(M)
        A = gen_action_set(M, act)
        p = env.execute_action(act)
        time += 1
        perception_ = perception
        perception = env.perception

        if env.flag:
            apply_alp(A, perception_, perception, time, Pop)
            # noinspection PyTypeChecker
            apply_rl(A, p)
            apply_ga(A, time, Pop)
        A_ = A
        if env.flag:
            env.stop()
            A_ = None

        if time % 100 == 0:
            perfClasseur.append(len(Pop))
            m = 0
            for cl in Pop:
                m += cl.q
            m /= len(Pop)
            meanQ.append(m)
            print(m)
        if time == 10000:
            break

    print(len(Pop))
    for cl in Pop:
        print('c : ', cl.condition, " a : ", cl.action, " e ", cl.effect, " q ", cl.q, " exp ", cl.exp,
              " age : ", cl.t)

    plt.plot(perfClasseur)
    plt.ylabel('Numbers of Classifier')
    plt.xlabel('Time')
    plt.show()

    plt.plot(meanQ)
    plt.ylabel('Quality q')
    plt.xlabel('Time')
    plt.show()

    print(meanQ)
    print(perfClasseur)
