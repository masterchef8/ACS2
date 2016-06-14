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

def remove(c: Classifier, set: list):
    """
    Remove this classifier from the current set
    :param c: Classifier to remove
    :param set:
    :return:
    """
    for i in range(len(set)):
        if c.condition == set[i].condition and c.action == set[i].action and c.effect == set[i].effect:
            del set[i]
            return True
    return False


def contains(setc: list, c: Classifier):
    """
    Check if the classifier is in the set
    :param setc:
    :param c:
    :return:
    """

    for i in range(len(setc)):
        if setc[i] == c:
            return True

    return False


def does_match(c, percept):
    """
    Check if classifier's condition match current perception
    :param c:
    :param percept:
    :return:
    """
    for i in range(len(percept)):
        if c.condition[i] != cons.symbol and c.condition[i] != percept[i]:
            return False
    return True


def gen_match_set(pop: list, percept: list):
    """
    Generate a list of Classifier thats match current perception
    :param pop:
    :type pop: list
    :param percept:
    :type percept: list
    :return:
    :rtype: list
    """
    ma = []
    if time == 0 or len(pop) == 0:
        for i in range(cons.nbAction):
            newcl = Classifier()
            newcl.condition = [cons.symbol] * cons.lenCondition
            newcl.action = i
            newcl.effect = [cons.symbol] * cons.lenCondition
            newcl.exp = 0
            newcl.t = time
            newcl.q = 0.5
            pop.append(newcl)
    for c in pop:
        if does_match(c, percept):
            ma.append(c)
    return ma


def update_application_average(cli: Classifier, t: int):
    """
    Update Classifier's parameters aav
    :type t: int
    :param t: Time
    :type cli: Classifier
    """
    if cli.exp < 1 / cons.beta:
        cli.aav += (t - cli.tga - cli.aav) / cli.exp
    else:
        cli.aav += cons.beta * (t - cli.tga - cli.aav)
    cli.tga = t


def does_anticipate_correctly(cla: Classifier, percept: list, percept_: list) -> bool:
    """

    :param cla: Classifier
    :param percept: Current perception
    :param percept_: Ancien perception
    :return:
    """
    for i in range(cons.lenCondition):
        if cla.effect == cons.symbol:
            if percept_[i] != percept[i]:
                return False
        else:
            if cla.effect[i] != percept[i] or percept_[i] == percept[i]:
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
            if int(mark[i]) > 1:
                type2 += 1

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


def expected_case(cli: Classifier, percept: list) -> Classifier:
    """

    :rtype: Classifier
    """
    diff = get_differences(cli.mark, percept)
    if diff == [cons.symbol] * cons.lenCondition:
        cli.q += cons.beta * (1 - cli.q)
        return None
    else:
        spec = number_of_spec(cli.condition)
        spec_new = number_of_spec(diff)
        child = Classifier(cli)
        if spec == cons.uMax:
            remove_random_spec_att(child.condition)
            spec -= 1
            while spec + spec_new > cons.beta:
                if spec > 0 and random() < 0.5:
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


def unexpected_case(clas: Classifier, percept: list, percept_: list) -> Classifier:
    """

    :rtype: Classifier
    """
    assert (len(percept_) == cons.lenCondition), "Wrong leight"
    assert (len(percept) == cons.lenCondition), "Wrong leight"

    clas.q = clas.q - cons.beta * clas.q
    clas.mark = percept_
    for i in range(len(percept)):
        if clas.effect[i] != cons.symbol:
            if clas.effect[i] != percept_[i] or percept_[i] != percept[i]:
                return None
    child = Classifier(clas)
    for i in range(len(percept)):
        if clas.effect[i] == cons.symbol and percept_[i] != percept[i]:
            child.condition[i] = percept_[i]
            child.effect[i] = percept[i]
    if clas.q < 0.5:
        clas.q = 0.5
    child.exp = 1
    return child


def add_alp_classifier(classifier: Classifier, pop: list, aset: list):
    oldcl = None
    for c in aset:
        if c.isSubsumer(classifier):
            if oldcl is None or c.isMoreGeneral(classifier):
                oldcl = c
    if oldcl is None:
        for c in aset:
            if c.equals(classifier):
                oldcl = c
    if oldcl is None:
        pop.append(classifier)
        aset.append(classifier)
    else:
        oldcl.q += cons.beta * (1 - oldcl.q)


def cover_triple(percept_: list, action: int, percept: list, t: int) -> Classifier:
    child = Classifier()

    for i in range(len(percept)):
        if percept_[i] != percept[i]:
            child.condition[i] = percept_[i]
            child.effect[i] = percept[i]
    child.action = action
    child.exp = 0
    child.r = 0
    child.aav = 0
    child.alp = t
    child.tga = t
    child.t = t
    child.q = 0.5
    child.num = 1
    return child


def apply_alp(actionset: list, percept_: list, percept: list, t: int, pop: list, action: int):
    """
    :param action:
    :param actionset:
    :type actionset: list
    :param percept_:
    :type percept_: list
    :param percept:
    :type percept: list
    :param t:
    :type t: int
    :param pop:
    :type pop: list
    """
    assert (len(percept_) == cons.lenCondition), "Wrong length"
    assert (len(percept) == cons.lenCondition), "Wrong length"

    wasExpectedCase = 0
    for c in actionset:
        contains(pop, c)
        c.exp += 1
        update_application_average(c, t)
        if does_anticipate_correctly(c, percept, percept_):
            newcl = expected_case(c, percept)
            wasExpectedCase += 1
        else:
            newcl = unexpected_case(c, percept, percept_)
            if c.q < cons.thetaI:
                # if c.gold is False:
                remove(c, Pop)
                actionset.remove(c)

        if newcl is not None:
            newcl.tga = t
            add_alp_classifier(newcl, pop, actionset)
    if wasExpectedCase == 0:
        newcl = cover_triple(percept_, action, percept, t)
        add_alp_classifier(newcl, pop, actionset)


def get_max_for_rl(matchset: list) -> int:
    """

    :param matchset:
    :return:
    """
    sumrl = 0
    for c in matchset:
        if c.effect != [cons.symbol] * cons.lenCondition:
            sumrl += c.q * c.r
    return sumrl


def apply_rl(aset: list, rwd: int):
    """

    :param aset:
    :param rwd:
    :param maxp:
    :type maxp: int
    """
    maxp = get_max_for_rl(M)
    for classifier in aset:
        classifier.r += cons.beta * (rwd + cons.gamma * maxp - classifier.r)
        classifier.ir += cons.beta * (rwd - classifier.ir)


def select_offspring(aset: list) -> Classifier:
    qualitySum = 0
    for cla in aset:
        qualitySum += cla.q ** 3
    choicePoint = random() * qualitySum
    qualitySum = 0
    for cla in aset:
        qualitySum += cla.q ** 3
        if qualitySum > choicePoint:
            return cla


def apply_ga_mutation(classifier: Classifier):
    for i in range(len(classifier.condition)):
        if classifier.condition[i] != cons.symbol:
            if random() < cons.mu:
                classifier.condition[i] = cons.symbol


def apply_crossover(cl1: Classifier, cl2: Classifier):
    if cl1.effect != cl2.effect:
        return
    x = random() * (len(cl1.condition) + 1)
    while True:
        y = random() * (len(cl1.condition) + 1)
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
        i += 1
        if i <= y:
            break


def delete_classifier(aset: list, pop: list):
    summation = 0
    for c in aset:
        summation += c.num
    while cons.inSize + summation > cons.thetaAS:
        cldel = None
        for classifier in pop:
            if random() < 1 / 3:
                if cldel is None:
                    cldel = classifier
                else:
                    if classifier.q - cldel.q < -0.1:
                        cldel = classifier
                    if abs(classifier.q - cldel.q) <= 0.1:
                        if classifier.mark is not None and cldel.mark is None:
                            cldel = classifier
                        elif classifier.mark is not None or cldel.mark is None:
                            if classifier.aav > cldel.aav:
                                cldel = classifier

        if cldel is not None:
            if cldel.num > 1:
                cldel.num -= 1
            else:
                pop.remove(classifier)
                remove(classifier, aset)
        summation = 0
        for c in aset:
            summation += c.num


def add_ga_classifier(aset: list, pop: list, classifier: Classifier):
    oldcl = None
    for c in aset:
        if c.isSubsumer(classifier):
            if oldcl is None or c.isMoreGeneral(classifier):
                oldcl = c
    if oldcl is None:
        for c in aset:
            if c.equals(classifier):
                oldcl = c
    if oldcl is None:
        pop.append(classifier)
        aset.append(classifier)
    else:
        if oldcl.mark is None:
            oldcl.num += 1


def apply_ga(aset: list, t: int, pop: list):
    sumNum = 0
    sumTgaN = 0
    for CL in aset:
        sumTgaN += CL.tga * CL.num
        sumNum += CL.num
    if (t - sumTgaN) / sumNum > cons.thetaGA:
        for CL in aset:
            CL.tga = t
        parent1 = select_offspring(aset)
        parent2 = select_offspring(aset)

        child1 = Classifier(parent1)
        child2 = Classifier(parent2)

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
        if child1.condition != [cons.symbol] * cons.lenCondition:
            add_ga_classifier(aset, pop, child1)

        if child2.condition != [cons.symbol] * cons.lenCondition:
            add_ga_classifier(aset, pop, child2)


def choose_action(aset: list):
    if random() < cons.epsilon:
        c = choice([i for i in range(cons.nbAction)])
        return c
    else:
        if len(aset) < 4:
            pass
        bestCL = aset[0]
        for classifierr in aset:
            if classifierr.effect != [cons.symbol] * cons.lenCondition and classifierr.q * classifierr.r > bestCL.q * bestCL.r:
                bestCL = classifierr
        return bestCL.action


def gen_action_set(aset: list, action: int):
    """

    :param action: action
    :type action: int
    :param aset: Match set
    :type aset: list
    :rtype: list
    """
    a = []

    for classifier in aset:
        if classifier.action == action:
            a.append(classifier)
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
    m = 0
    perception = env.perception
    while True:

        M = gen_match_set(Pop, perception)
        if len(Pop) < 4:
            pass
        if A_ is not None:
            apply_alp(A_, perception_, perception, time, Pop, act)
            apply_rl(A_, p)
            apply_ga(A_, time, Pop)
        A_ = None
        act = choose_action(M)
        A = gen_action_set(M, act)
        p = env.execute_action(act)
        time += 1
        perception_ = perception
        perception = env.perception
        A_ = A
        if env.flag:

            apply_alp(A, perception_, perception, time, Pop, act)

            apply_rl(A, p)

            apply_ga(A, time, Pop)


        if env.flag:
            env.stop()
            A_ = None
        # ------------------------------------------------------

        if time % 100 == 0:
            perfClasseur.append(len(Pop))
            m = 0
            for cl in Pop:
                m += cl.q
            m /= len(Pop)
            meanQ.append(m)
            print(m, len(Pop))
            print(time)
        if time == 30000:
            break

    print(len(Pop))
    for cl in Pop:
        print(cl.condition, cl.action, cl.effect, cl.q, cl.t)

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
