# coding=utf-8
"""
@author : somebody
@date : 04/06/16

@project : ACS2
@file : Classifier

@Class description :
"""

import Constant

cons = Constant.Constants()


class Classifier(object):
    def __init__(self, copy=None):

        self.condition = [cons.symbol] * cons.lenCondition

        self.action = None

        self.effect = [cons.symbol] * cons.lenCondition
        self.q = 0.5  # quality
        self.r = 0  # reward

        self.mark = None  # All cases were effect was not good
        self.ir = 0  # Immediate reward
        self.t = None  # Has been created at this time
        self.tga = 0  # Last time that self was part of action set with GA
        self.alp = 0  # Last time that self underwent ALP update

        self.aav = 0  # application average
        self.exp = 0  # experience
        self.num = 1  # Still Micro or macro stuffs
        self.gold = False
        if self.condition == [cons.symbol] * cons.lenCondition:
            if self.effect == [cons.symbol] * cons.lenCondition:
                if self.t == 0:
                    self.gold = True
        else:
            self.gold = False

        if copy is not None:
            assert type(copy) is Classifier, "Wrong type for copy"

            self.condition = copy.condition

            self.action = copy.action

            self.effect = copy.effect
            self.q = copy.q  # quality
            self.r = copy.r  # reward

            self.mark = copy.mark  # All cases were effect was not good
            self.ir = copy.ir  # Immediate reward
            self.t = copy.t  # Has been created at this time
            self.tga = copy.tga  # Last time that self was part of action set with GA
            self.alp = copy.alp  # Last time that self underwent ALP update

            self.aav = copy.aav  # application average
            self.exp = copy.exp  # experience
            self.num = copy.num  # Still Micro or macro stuffs

    def isSubsumer(self, cl):
        """

        :type cl: Classifier
        :rtype: bool
        """
        assert (type(cl) is Classifier), "Bad type for cl in function Classifier.isSubsumer"
        cp = 0
        cpt = 0
        if self.exp > cons.thetaExp and self.q > cons.thetaR and self.mark is None:
            for i in range(cons.lenCondition):
                if self.condition[i] == cons.symbol:
                    cp += 1
                if cl.condition[i] == cons.symbol:
                    cpt += 1
            if cp <= cpt:
                if self.effect == cl.effect:
                    return True
        return False

    def isMoreGeneral(self, cl):
        """ Returns if the classifier is more general than cl. It is made sure that the classifier is indeed more general and
        not equally general as well as that the more specific classifier is completely included in the more general one (do not specify overlapping regions)
        :param cl: The classifier that is tested to be more specific. """

        ret = False
        length = len(self.condition)
        for i in range(length):
            if self.condition[i] != cons.symbol and self.condition[i] != cl.condition[i]:
                return False
            elif self.condition[i] != cl.condition[i]:
                ret = True
        return ret

    def equals(self, cl):
        """ Returns if the two classifiers are identical in condition and action.
        @param cl The classifier to be compared. """

        if cl.condition == self.condition:
            if cl.action == self.action:
                return True
        return False
