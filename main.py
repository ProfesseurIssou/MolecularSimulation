import numpy as np
from numba import njit
import pygame, time


WINDOW_SIZE = 1000

MOLECULES_RADIUS = 5

#### RULES - NUCLEUS ####
NB_RED_MOLECULES = 200
NB_GREEN_MOLECULES = 200
NB_BLUE_MOLECULES = 0
NB_YELLOW_MOLECULES = 200

MOLECULES_FORCE_RANGE_RED_RED = 160
MOLECULES_FORCE_RANGE_RED_GREEN = 160
MOLECULES_FORCE_RANGE_RED_BLUE = 160
MOLECULES_FORCE_RANGE_RED_YELLOW = 160

MOLECULES_FORCE_RANGE_GREEN_RED = 160
MOLECULES_FORCE_RANGE_GREEN_GREEN = 160
MOLECULES_FORCE_RANGE_GREEN_BLUE = 160
MOLECULES_FORCE_RANGE_GREEN_YELLOW = 160

MOLECULES_FORCE_RANGE_BLUE_RED = 160
MOLECULES_FORCE_RANGE_BLUE_GREEN = 160
MOLECULES_FORCE_RANGE_BLUE_BLUE = 160
MOLECULES_FORCE_RANGE_BLUE_YELLOW = 160

MOLECULES_FORCE_RANGE_YELLOW_RED = 160
MOLECULES_FORCE_RANGE_YELLOW_GREEN = 160
MOLECULES_FORCE_RANGE_YELLOW_BLUE = 160
MOLECULES_FORCE_RANGE_YELLOW_YELLOW = 160

RULE_RED_RED = 0.1
RULE_RED_GREEN = -0.1
RULE_RED_BLUE = 0
RULE_RED_YELLOW = 0

RULE_GREEN_RED = -0.2
RULE_GREEN_GREEN = -0.7
RULE_GREEN_BLUE = 0
RULE_GREEN_YELLOW = 0

RULE_BLUE_RED = 0
RULE_BLUE_GREEN = 0
RULE_BLUE_BLUE = 0
RULE_BLUE_YELLOW = 0

RULE_YELLOW_RED = 0.15
RULE_YELLOW_GREEN = 0
RULE_YELLOW_BLUE = 0
RULE_YELLOW_YELLOW = 0
#########################

# #### RULES - EATER ####
# NB_RED_MOLECULES = 200
# NB_GREEN_MOLECULES = 200
# NB_BLUE_MOLECULES = 0
# NB_YELLOW_MOLECULES = 200

# MOLECULES_FORCE_RANGE_RED_RED = 160
# MOLECULES_FORCE_RANGE_RED_GREEN = 160
# MOLECULES_FORCE_RANGE_RED_BLUE = 160
# MOLECULES_FORCE_RANGE_RED_YELLOW = 160

# MOLECULES_FORCE_RANGE_GREEN_RED = 160
# MOLECULES_FORCE_RANGE_GREEN_GREEN = 160
# MOLECULES_FORCE_RANGE_GREEN_BLUE = 160
# MOLECULES_FORCE_RANGE_GREEN_YELLOW = 160

# MOLECULES_FORCE_RANGE_BLUE_RED = 160
# MOLECULES_FORCE_RANGE_BLUE_GREEN = 160
# MOLECULES_FORCE_RANGE_BLUE_BLUE = 160
# MOLECULES_FORCE_RANGE_BLUE_YELLOW = 160

# MOLECULES_FORCE_RANGE_YELLOW_RED = 160
# MOLECULES_FORCE_RANGE_YELLOW_GREEN = 160
# MOLECULES_FORCE_RANGE_YELLOW_BLUE = 160
# MOLECULES_FORCE_RANGE_YELLOW_YELLOW = 160

# RULE_RED_RED = -0.1
# RULE_RED_GREEN = -0.34
# RULE_RED_BLUE = 0
# RULE_RED_YELLOW = 0

# RULE_GREEN_RED = -0.17
# RULE_GREEN_GREEN = -0.32
# RULE_GREEN_BLUE = 0
# RULE_GREEN_YELLOW = 0.34

# RULE_BLUE_RED = 0
# RULE_BLUE_GREEN = 0
# RULE_BLUE_BLUE = 0
# RULE_BLUE_YELLOW = 0

# RULE_YELLOW_RED = 0
# RULE_YELLOW_GREEN = -0.2
# RULE_YELLOW_BLUE = 0
# RULE_YELLOW_YELLOW = 0.15
# ########################

#### RULES ####
# NB_RED_MOLECULES = 1000
# NB_GREEN_MOLECULES = 1000
# NB_BLUE_MOLECULES = 730
# NB_YELLOW_MOLECULES = 1000

# MOLECULES_FORCE_RANGE_RED_RED = 200
# MOLECULES_FORCE_RANGE_RED_GREEN = 270
# MOLECULES_FORCE_RANGE_RED_BLUE = 240
# MOLECULES_FORCE_RANGE_RED_YELLOW = 200

# MOLECULES_FORCE_RANGE_GREEN_RED = 200
# MOLECULES_FORCE_RANGE_GREEN_GREEN = 345
# MOLECULES_FORCE_RANGE_GREEN_BLUE = 200
# MOLECULES_FORCE_RANGE_GREEN_YELLOW = 200

# MOLECULES_FORCE_RANGE_BLUE_RED = 71
# MOLECULES_FORCE_RANGE_BLUE_GREEN = 230
# MOLECULES_FORCE_RANGE_BLUE_BLUE = 81
# MOLECULES_FORCE_RANGE_BLUE_YELLOW = 365

# MOLECULES_FORCE_RANGE_YELLOW_RED = 360
# MOLECULES_FORCE_RANGE_YELLOW_GREEN = 315
# MOLECULES_FORCE_RANGE_YELLOW_BLUE = 200
# MOLECULES_FORCE_RANGE_YELLOW_YELLOW = 200

# RULE_RED_RED = 70
# RULE_RED_GREEN = -20
# RULE_RED_BLUE = 31.5
# RULE_RED_YELLOW = 0

# RULE_GREEN_RED = 0
# RULE_GREEN_GREEN = 16.5
# RULE_GREEN_BLUE = 0
# RULE_GREEN_YELLOW = 0

# RULE_BLUE_RED = 1.5
# RULE_BLUE_GREEN = 15.5
# RULE_BLUE_BLUE = -13.5
# RULE_BLUE_YELLOW = 2.5

# RULE_YELLOW_RED = 25
# RULE_YELLOW_GREEN = 15
# RULE_YELLOW_BLUE = 0
# RULE_YELLOW_YELLOW = -45
###############


@njit(cache = True)
def ProcessMolecules(
    redMolecules: np.ndarray,
    greenMolecules: np.ndarray,
    blueMolecules: np.ndarray,
    yellowMolecules: np.ndarray,

    redMoleculesSpeed: np.ndarray,
    greenMoleculesSpeed: np.ndarray,
    blueMoleculesSpeed: np.ndarray,
    yellowMoleculesSpeed: np.ndarray,
    ) -> (np.ndarray):
    """
        Process the molecules and return the new positions and speeds
    """

    # Update the positions
    redMolecules += redMoleculesSpeed
    greenMolecules += greenMoleculesSpeed
    blueMolecules += blueMoleculesSpeed
    yellowMolecules += yellowMoleculesSpeed

    # Limit the positions, reverse speed if collision with the wall
    for i in range(redMolecules.shape[0]):
        if redMolecules[i, 0] < 0 or redMolecules[i, 0] > WINDOW_SIZE:
            redMoleculesSpeed[i, 0] *= -1
        if redMolecules[i, 1] < 0 or redMolecules[i, 1] > WINDOW_SIZE:
            redMoleculesSpeed[i, 1] *= -1            
    for i in range(greenMolecules.shape[0]):
        if greenMolecules[i, 0] < 0 or greenMolecules[i, 0] > WINDOW_SIZE:
            greenMoleculesSpeed[i, 0] *= -1
        if greenMolecules[i, 1] < 0 or greenMolecules[i, 1] > WINDOW_SIZE:
            greenMoleculesSpeed[i, 1] *= -1
    for i in range(blueMolecules.shape[0]):
        if blueMolecules[i, 0] < 0 or blueMolecules[i, 0] > WINDOW_SIZE:
            blueMoleculesSpeed[i, 0] *= -1
        if blueMolecules[i, 1] < 0 or blueMolecules[i, 1] > WINDOW_SIZE:
            blueMoleculesSpeed[i, 1] *= -1
    for i in range(yellowMolecules.shape[0]):
        if yellowMolecules[i, 0] < 0 or yellowMolecules[i, 0] > WINDOW_SIZE:
            yellowMoleculesSpeed[i, 0] *= -1
        if yellowMolecules[i, 1] < 0 or yellowMolecules[i, 1] > WINDOW_SIZE:
            yellowMoleculesSpeed[i, 1] *= -1

    # Update the speeds
    redMoleculesSpeed = UpdateSpeeds(redMolecules, redMoleculesSpeed, redMolecules, RULE_RED_RED, MOLECULES_FORCE_RANGE_RED_RED)
    redMoleculesSpeed = UpdateSpeeds(redMolecules, redMoleculesSpeed, greenMolecules, RULE_RED_GREEN, MOLECULES_FORCE_RANGE_RED_GREEN)
    redMoleculesSpeed = UpdateSpeeds(redMolecules, redMoleculesSpeed, blueMolecules, RULE_RED_BLUE, MOLECULES_FORCE_RANGE_RED_BLUE)
    redMoleculesSpeed = UpdateSpeeds(redMolecules, redMoleculesSpeed, yellowMolecules, RULE_RED_YELLOW, MOLECULES_FORCE_RANGE_RED_YELLOW)

    greenMoleculesSpeed = UpdateSpeeds(greenMolecules, greenMoleculesSpeed, redMolecules, RULE_GREEN_RED, MOLECULES_FORCE_RANGE_GREEN_RED)
    greenMoleculesSpeed = UpdateSpeeds(greenMolecules, greenMoleculesSpeed, greenMolecules, RULE_GREEN_GREEN, MOLECULES_FORCE_RANGE_GREEN_GREEN)
    greenMoleculesSpeed = UpdateSpeeds(greenMolecules, greenMoleculesSpeed, blueMolecules, RULE_GREEN_BLUE, MOLECULES_FORCE_RANGE_GREEN_BLUE)
    greenMoleculesSpeed = UpdateSpeeds(greenMolecules, greenMoleculesSpeed, yellowMolecules, RULE_GREEN_YELLOW, MOLECULES_FORCE_RANGE_GREEN_YELLOW)

    blueMoleculesSpeed = UpdateSpeeds(blueMolecules, blueMoleculesSpeed, redMolecules, RULE_BLUE_RED, MOLECULES_FORCE_RANGE_BLUE_RED)
    blueMoleculesSpeed = UpdateSpeeds(blueMolecules, blueMoleculesSpeed, greenMolecules, RULE_BLUE_GREEN, MOLECULES_FORCE_RANGE_BLUE_GREEN)
    blueMoleculesSpeed = UpdateSpeeds(blueMolecules, blueMoleculesSpeed, blueMolecules, RULE_BLUE_BLUE, MOLECULES_FORCE_RANGE_BLUE_BLUE)
    blueMoleculesSpeed = UpdateSpeeds(blueMolecules, blueMoleculesSpeed, yellowMolecules, RULE_BLUE_YELLOW, MOLECULES_FORCE_RANGE_BLUE_YELLOW)

    yellowMoleculesSpeed = UpdateSpeeds(yellowMolecules, yellowMoleculesSpeed, redMolecules, RULE_YELLOW_RED, MOLECULES_FORCE_RANGE_YELLOW_RED)
    yellowMoleculesSpeed = UpdateSpeeds(yellowMolecules, yellowMoleculesSpeed, greenMolecules, RULE_YELLOW_GREEN, MOLECULES_FORCE_RANGE_YELLOW_GREEN)
    yellowMoleculesSpeed = UpdateSpeeds(yellowMolecules, yellowMoleculesSpeed, blueMolecules, RULE_YELLOW_BLUE, MOLECULES_FORCE_RANGE_YELLOW_BLUE)
    yellowMoleculesSpeed = UpdateSpeeds(yellowMolecules, yellowMoleculesSpeed, yellowMolecules, RULE_YELLOW_YELLOW, MOLECULES_FORCE_RANGE_YELLOW_YELLOW)
    
    return redMolecules, greenMolecules, blueMolecules, yellowMolecules, redMoleculesSpeed, greenMoleculesSpeed, blueMoleculesSpeed, yellowMoleculesSpeed

@njit
def UpdateSpeeds(
    molecules: np.ndarray,
    moleculesSpeed: np.ndarray,
    otherMolecules: np.ndarray,
    rule: float,
    forceRange: float
    ) -> np.ndarray:
    """
        Update the speed of the molecules
    """
    for i in range(len(molecules)):
        forceX = 0.0
        forceY = 0.0
        for j in range(len(otherMolecules)):
            distance = molecules[i] - otherMolecules[j]
            distX = distance[0]
            distY = distance[1]
            dist = np.sqrt(distX**2 + distY**2)
            if dist != 0.0 and dist < forceRange:
                force = rule * (1/dist)
                forceX += force * distX
                forceY += force * distY
        moleculesSpeed[i] = (moleculesSpeed[i]+np.array([forceX, forceY])) * 0.95 # Plus il est grand, plus les molécules vont vite (et plus elles vont se repousser)
    return moleculesSpeed



#Creation des listes de molecules
redMoleculesArray = np.zeros((NB_RED_MOLECULES, 2), dtype=np.float32)
greenMoleculesArray = np.zeros((NB_GREEN_MOLECULES, 2), dtype=np.float32)
blueMoleculesArray = np.zeros((NB_BLUE_MOLECULES, 2), dtype=np.float32)
yellowMoleculesArray = np.zeros((NB_YELLOW_MOLECULES, 2), dtype=np.float32)

#Creation des listes de vitesses
redSpeedsArray = np.zeros((NB_RED_MOLECULES, 2), dtype=np.float32)
greenSpeedsArray = np.zeros((NB_GREEN_MOLECULES, 2), dtype=np.float32)
blueSpeedsArray = np.zeros((NB_BLUE_MOLECULES, 2), dtype=np.float32)
yellowSpeedsArray = np.zeros((NB_YELLOW_MOLECULES, 2), dtype=np.float32)

#Generation des positions aleatoires
redMoleculesArray[:, 0] = np.random.randint(0, WINDOW_SIZE, NB_RED_MOLECULES)
redMoleculesArray[:, 1] = np.random.randint(0, WINDOW_SIZE, NB_RED_MOLECULES)
greenMoleculesArray[:, 0] = np.random.randint(0, WINDOW_SIZE, NB_GREEN_MOLECULES)
greenMoleculesArray[:, 1] = np.random.randint(0, WINDOW_SIZE, NB_GREEN_MOLECULES)
blueMoleculesArray[:, 0] = np.random.randint(0, WINDOW_SIZE, NB_BLUE_MOLECULES)
blueMoleculesArray[:, 1] = np.random.randint(0, WINDOW_SIZE, NB_BLUE_MOLECULES)
yellowMoleculesArray[:, 0] = np.random.randint(0, WINDOW_SIZE, NB_YELLOW_MOLECULES)
yellowMoleculesArray[:, 1] = np.random.randint(0, WINDOW_SIZE, NB_YELLOW_MOLECULES)



pygame.init()
window = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
loop = True
while (loop) :
    window.fill((0, 0, 0))
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            loop = False



    #Update the molecules
    (
        redMoleculesArray, 
        greenMoleculesArray, 
        blueMoleculesArray, 
        yellowMoleculesArray, 
        redSpeedsArray, 
        greenSpeedsArray, 
        blueSpeedsArray, 
        yellowSpeedsArray
    ) = ProcessMolecules(
        redMoleculesArray, 
        greenMoleculesArray, 
        blueMoleculesArray, 
        yellowMoleculesArray, 
        redSpeedsArray, 
        greenSpeedsArray, 
        blueSpeedsArray, 
        yellowSpeedsArray
    )


    #Affichage des molecules
    for i in range(NB_RED_MOLECULES):
        pygame.draw.circle(window, (255, 0, 0), (int(redMoleculesArray[i, 0]), int(redMoleculesArray[i, 1])), MOLECULES_RADIUS)
    for i in range(NB_GREEN_MOLECULES):
        pygame.draw.circle(window, (0, 255, 0), (int(greenMoleculesArray[i, 0]), int(greenMoleculesArray[i, 1])), MOLECULES_RADIUS)
    for i in range(NB_BLUE_MOLECULES):
        pygame.draw.circle(window, (0, 0, 255), (int(blueMoleculesArray[i, 0]), int(blueMoleculesArray[i, 1])), MOLECULES_RADIUS)
    for i in range(NB_YELLOW_MOLECULES):
        pygame.draw.circle(window, (255, 255, 0), (int(yellowMoleculesArray[i, 0]), int(yellowMoleculesArray[i, 1])), MOLECULES_RADIUS)
    pygame.display.flip()

    #60 fps
    # pygame.time.wait(16)