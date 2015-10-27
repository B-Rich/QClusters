"""
This script simulate different kinds of mesoscopic quadropole clusters melting.
Initial particles positions must loaded from file 'quadropoles/eq**.txt'.
They are created by gradient descent method.
Script 'PostProduction.py' draws result of simulaton.
More info about clusters melting at:
1. http://journals.ioffe.ru/ftt/1999/08/p1499-1504.pdf
2. http://journals.ioffe.ru/ftt/1998/07/p1379-1386.pdf
"""

from numpy import *
from sys import maxint
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import time

SpeedStep = 0.035
TimeSteps = 3000
Thermalization = 300

TimeStepLen = 0.012
TempSteps = 15

OutRadius = [1.7, 3.0]
ObsRadius = [0.9, 1.7]

MaxSpeed = 0.03
ParticleMass = 1.0

rareFactor = 1.0
frameSize = 3.5

source = 'quadropoles/eq20.txt'
reportsPath = "statistics/"
rsFilename = reportsPath + "radiuses"
asFilename = reportsPath + "angles"

def cart2pol(x, y):
    rho = sqrt(x**2 + y**2)
    phi = arctan2(y, x)
    return(rho, phi)

def pol2cart(rho, phi):
    x = rho * cos(phi)
    y = rho * sin(phi)
    return(x, y)

def touch(path):
    with open(path, 'a'):
        os.utime(path, None)

def get_draw_context():
    mpl.rcParams['toolbar'] = 'None'
    fig = plt.figure()
    axes = plt.gca()
    ax = fig.add_subplot(111)
    return fig, ax

def savePoint(filename, x, y, xErr=0.0, yErr=0.0):
    with open(filename, 'a') as file:
        file.write("{0} {1} {2} {3}\n".format(x, y, xErr, yErr))

def clearFile(filename):
    open(filename, 'w').close()

def dotProduct(v1, v2):
    return sum((a*b) for a, b in zip(v1, v2))

def vecLength(v):
    return sqrt(dotProduct(v, v))

def angleBv(v1, v2):
    cos = dotProduct(v1, v2) / (vecLength(v1) * vecLength(v2))
    if -1.0 <= cos <= 1.0:
        return arccos(dotProduct(v1, v2) / (vecLength(v1) * vecLength(v2)))
    else:
        return pi if cos > 1.0 else -pi

def angleList(XPositions, YPositions):
    axis = [1, 0]
    angles = []
    for xc, yc in zip(XPositions, YPositions):
        vector = [xc, yc]
        angles.append(angleBv(axis, vector))
    return angles

##################################################################################
##################################################################################

def electicalForce(XPositions, YPositions, pIndex):
    posMatrix = vstack([XPositions[pIndex] - XPositions,
                         YPositions[pIndex] - YPositions])
    modules = linalg.norm(posMatrix, axis = 0)
    modules[pIndex] = inf
    forceMatrix = -1.0 * 5 * posMatrix / modules ** 7
    return [sum(forceMatrix[0, :]), sum(forceMatrix[1, :])]

def potentialForce(XPositions, YPositions, particleIndex):
    return [2.0 * XPositions[particleIndex], 2.0 * YPositions[particleIndex]]

class crystal:
    if __name__ == "__main__": fig, ax = get_draw_context()

    def __init__(self, filename, forceFunction):
        self.positions = empty((0,2))
        self.forces = forceFunction
        self.firstFrame = True

        with open(filename,'r') as file:
            for line in file:
                xc, yc = line.split(" ")
                self.positions = vstack([self.positions, [float(xc), float(yc)]])

        self.positions = self.positions.T
        self.pointsCount = size(self.positions[0])
        self.forceMatrix = zeros((2, self.pointsCount), float)

    def initialiseSpeeds(self, maxSpeed):
        self.speeds = 2 * maxSpeed * random.rand(2, self.pointsCount) - maxSpeed

    def crystalShell(self, minDistance, maxDistance):
        distance = linalg.norm(self.positions, axis = 0)
        indexes = []
        distance = array(distance)
        for j in range(len(distance)):
            if minDistance <= distance[j] <= maxDistance:
                indexes.append(j)
        return indexes

    def closestShellNeighbors(self, shell):
        particlesCount = len(self.positions[0])
        metrics = lambda x0, y0, x1, y1: sqrt(pow(x0 - x1, 2) + pow(y0 - y1, 2))
        cnDict = {}
        for i in range(particlesCount):
            if i in shell and not i in cnDict.values():
                minDistance = maxint
                cnIndex = -1
                for j in range(particlesCount):
                    if j in shell and i != j:
                        distance = metrics(self.positions[0, i], self.positions[1, i],
                                           self.positions[0, j], self.positions[1, j])
                        if distance < minDistance:
                            minDistance = distance
                            cnIndex = j
                cnDict[i] = cnIndex
        return cnDict

    def averageDistance(self, pairs):
        distance_list = []
        for j in range(len(pairs)):
            dictValue = pairs.keys()[j]
            fIndex, sIndex = dictValue, pairs[dictValue]
            fvector = [self.positions[0, fIndex], self.positions[1, fIndex]]
            svector = [self.positions[0, sIndex], self.positions[1, sIndex]]
            distance = sqrt(pow(fvector[0] - svector[0], 2) + pow(fvector[1] - svector[1], 2))
            distance_list.append(distance)

        average = sum(distance_list)/len(distance_list)
        return average

    def drawCrystal(self, frameSize, shell):
        if self.firstFrame:
            fig = plt.gcf()
            fig.canvas.set_window_title('Cluster')
            plt.xlim(-frameSize, frameSize)
            plt.ylim(-frameSize, frameSize)
            plt.ion()
            plt.show()
            self.firstFrame = False
        sl = []
        for j in range(self.pointsCount):
            if j in shell:
                sl.append(crystal.ax.scatter([self.positions[0, j]],
                            [self.positions[1, j]], color = "r", edgecolors='w', s=60, alpha=0.7))
            else:
                sl.append(crystal.ax.scatter([self.positions[0, j]],
                            [self.positions[1, j]], color = "b", edgecolors='w', s=60, alpha=0.7))
        crystal.fig.canvas.draw()
        for j in range(len(sl)):
            sl[j].remove()

    def increaseSpeeds(self, degree):
        modules = [sqrt(square(self.speeds[0, j]) + square(self.speeds[1, j]))
                   for j in range(self.pointsCount)]
        meanSpeed = mean(modules)
        alpha = 1.0 + degree / meanSpeed
        self.speeds *= alpha

    def temperature(self):
        return ParticleMass * (sum(power(self.speeds[0], 2)) +
                                sum(power(self.speeds[1], 2))) / (2.0 * self.pointsCount)

    def interShellNeighbors(self, Shell, nShell):
        module = lambda firstPoint, secondPoint: sqrt((firstPoint[0] - secondPoint[0])**2 +
                                                             (firstPoint[1] - secondPoint[1])**2)
        cnDict = {}
        for i in range(len(Shell)):
            minDistance = 100500.0
            for j in range(len(nShell)):
                Distance = module([self.positions[0, Shell[i]], self.positions[1, Shell[i]]],
                                  [self.positions[0, nShell[j]], self.positions[1, nShell[j]]])
                if Distance < minDistance:
                    minDistance = Distance
                    cnDict[Shell[i]] = nShell[j]
        return cnDict

    def shellOrientation(self, neighborsDict):
        angles = []
        for key, value in neighborsDict.iteritems():
            first_vector = [self.positions[0, key], self.positions[1, key]]
            second_vector = [self.positions[0, value], self.positions[1, value]]
            angles.append(angleBv(first_vector, second_vector) * 180 / pi)
        return angles

    def particleAngles(self, observedShell):
        angles = []
        for index in observedShell:
            rho, phi = cart2pol(self.positions[0, index], self.positions[1, index])
            angles.append(phi * 180 / pi)
        return angles

    def particleModules(self, observedShell):
        return linalg.norm(vstack([self.positions[[0], observedShell],
                                   self.positions[[1], observedShell]]), axis = 0)

    def computeForces(self):
        forceMatrix = empty((0, 2))
        for i in xrange(self.pointsCount):
            forceMatrix = vstack([forceMatrix, self.forces(self.positions[0], self.positions[1], i)])
        return forceMatrix.T

    def leapFrogStep(self):
        nForceMatrix = self.computeForces()
        nSpeeds = self.speeds - TimeStepLen * (nForceMatrix + self.forceMatrix) / (2.0 * ParticleMass)
        nPositions = self.positions + nSpeeds * TimeStepLen - self.forceMatrix * (TimeStepLen ** 2) / 2.0

        self.forceMatrix = copy(nForceMatrix)
        self.positions = copy(nPositions)
        self.speeds = copy(nSpeeds)

    def showMelting(self, step, selectedPoints, passSteps = 250):
        if step % passSteps == 0:
            self.drawCrystal(frameSize, selectedPoints)
            print step

    def meltingStat(self, mode, steps, observedPoints = [], pairs = [], neighborPoints = [], drawObject = True):
        radiuses = empty((0, len(observedPoints)), float)
        nsAngles = empty((0, len(pairs)), float)
        sAngles = empty((0, len(neighborPoints)), float)
        self.tempEvolution = array([])

        for k in range(steps):
            self.leapFrogStep()
            if drawObject: self.showMelting(k, observedPoints)
            if k > Thermalization:
                self.tempEvolution = append(self.tempEvolution, self.temperature())
                # distances from center
                if mode == "rd":
                    radiuses = vstack([radiuses, self.particleModules(observedPoints)])
                # intershell angles
                elif mode == "is":
                    nsAngles = vstack([nsAngles, self.shellOrientation(pairs)])
                # shell angles
                elif mode == "sa":
                    sAngles = vstack([sAngles, self.particleAngles(neighborPoints)])

        if mode == "rd": return radiuses, mean(self.tempEvolution)
        elif mode == "is": return nsAngles, mean(self.tempEvolution)
        elif mode == "sa": return sAngles, mean(self.tempEvolution)

def calculateDeltaR(distance, iterationsCount, lattice_parameter, shellSize):
    averageR = []
    averageR2 = []
    for p in range(shellSize):
        averageR.append(sum(distance[:,p]) / (rareFactor * iterationsCount))
        averageR2.append(sum(power(distance[:,p], 2)) / (rareFactor * iterationsCount))
    return sum(subtract(averageR2, power(averageR, 2)))/(shellSize * lattice_parameter ** 2)

def calculateDeltaPhi(angles, iterationsCount, PhiZero, shellSize):
    averageA = array([])
    averageA2 = array([])
    for p in range(shellSize):
        averageA = append(averageA, (sum(angles[:,p]) / (rareFactor * iterationsCount)))
        averageA2 = append(averageA2, (sum(power(angles[:,p], 2)) / (rareFactor * iterationsCount)))
    return sum(subtract(averageA2, power(averageA, 2)))/ (shellSize * PhiZero ** 2)

def refreshReports():
    clearFile(rsFilename) if os.path.isfile(rsFilename) else touch(rsFilename)
    clearFile(asFilename) if os.path.isfile(asFilename) else touch(asFilename)

def saveDisturb(name, lst, blocksNumber = 100):
    hist, bins = histogram(lst, blocksNumber)
    shist = []
    for n in range(2, len(hist) - 2):
        shist.append((0.15 * hist[n-2] + 0.20 * hist[n-1] +
                      0.3 * hist[n] + 0.20 * hist[n+1] +
                      0.15 * hist[n+2]) / 5.0)
    print shist
    shist = array(shist)
    shist /= sum(shist)
    center = (bins[:-1] + bins[1:]) / 2
    center = list(center[2:-2])
    plt.clf()

    plt.plot(center, shist)
    for x, y in zip(center, shist):
        savePoint("{0}{1}.txt".format(reportsPath, name), x, y)

    plt.savefig("{0}{1}.png".format(reportsPath, name))

##################################################################################
##################################################################################

def ultimateMeltingDemo():
    # At mean square radius plot you should see jump of the value.
    # For better results quality you must increase timesteps,
    # but in other hand calculation time increases too. You are
    # welcome to experiment!
    SpeedStep = 0.035
    MaxSpeed = 0.03
    TimeSteps = 3000
    TimeStepLen = 0.012

    cluster.initialiseSpeeds(MaxSpeed)
    pairs = cluster.interShellNeighbors(outShell, shell)
    latticeParam = cluster.averageDistance(pairs)

    for step in xrange(TempSteps):
        print "point #" + str(step)
        if step > 0: cluster.increaseSpeeds(SpeedStep)
        distance, temperature = cluster.meltingStat('rd', TimeSteps + Thermalization,
                                                    observedPoints = shell)
        deltaR = calculateDeltaR(distance, TimeSteps, latticeParam, len(shell))
        savePoint(rsFilename, temperature, deltaR)

def orientMeltingDemo():
    # At mean square angle plot you should see jump of the value too.
    # Initial speed and speed step must be very small.
    SpeedStep = 0.005
    MaxSpeed = 0.006
    TimeSteps = 3000
    TimeStepLen = 0.012

    cluster.initialiseSpeeds(MaxSpeed)
    sPairs = cluster.closestShellNeighbors(shell)
    phiZero = 360.0 / len(shell)

    for step in xrange(TempSteps):
        print "point #" + str(step)
        if step > 0: cluster.increaseSpeeds(SpeedStep)
        isAngles, temperature = cluster.meltingStat('is', TimeSteps + Thermalization,
                                                    pairs = sPairs)
        deltaPhi = calculateDeltaPhi(isAngles, TimeSteps, phiZero, len(sPairs))
        savePoint(asFilename, temperature, deltaPhi)

def disturbDemo():
    SpeedStep = 0.01
    MaxSpeed = 0.01

    cluster.initialiseSpeeds(MaxSpeed)
    angles = []
    pairs = cluster.interShellNeighbors(outShell, shell)
    for step in xrange(TempSteps):
        print "point #" + str(step)
        if step > 0: cluster.increaseSpeeds(SpeedStep)
        if step == 5:
            angles, temperature = cluster.meltingStat('sa', 1000,
                                                        neighborPoints = range(len(cluster.speeds[0])),
                                                        drawObject = False)
            break
        else:
            angles, temperature = cluster.meltingStat('sa', 1000,
                                                        neighborPoints = range(len(cluster.speeds[0])),
                                                        drawObject = False)
    saveDisturb("angle", angles, 200)


if __name__ == "__main__":

    forces = lambda XPos, YPos, index: add(electicalForce(XPos, YPos, index),
                                           potentialForce(XPos, YPos, index))

    cluster = crystal(source, forces)

    refreshReports()

    shell = cluster.crystalShell(ObsRadius[0], ObsRadius[1])
    outShell = cluster.crystalShell(OutRadius[0], OutRadius[1])

    # Try to uncomment one of these lines
    disturbDemo()
    #orientMeltingDemo()
    #ultimateMeltingDemo()






