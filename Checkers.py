import copy
import random
import sys
import pygame

ROWS, COLS=8, 8

# Define constants
WIDTH, HEIGHT = 500, 500
GRID_SIZE = WIDTH // 8
WHITE = (0, 0,0)
BLACK = (0, 0, 0)
RED = (0, 0, 255)
BLUE = (255, 0, 0)
GREEN = (255, 255, 0)
YELLOW = (0, 255, 0)
CROWN = pygame.transform.scale(pygame.image.load('crown.png'), (44, 25))

COLOR1 = (255, 255, 255)
COLOR2 = (0, 0, 0)

black_dx = [+1, +1]
black_dy = [-1, +1]

red_dx = [-1, -1]
red_dy = [-1, +1]

king_dx = [-1, -1, +1, +1]
king_dy = [-1, +1, -1, +1]

class ExperimentGenerator:
    def __init__(self):
        self.board = self.generateBoard()
        self.history = [copy.deepcopy(self.board)]
        self.numMoves = 0

    def setBoard(self,board):
        if board == 0:
            print ("zero board")
        self.board = board
        self.history.append(copy.deepcopy(self.board))

    def generateBoard(self):
        # 1=black piece 2= red piece -1 = black king -2 = red king 
        board=[[0,1,0,1,0,1,0,1],
               [1,0,1,0,1,0,1,0],
               [0,1,0,1,0,1,0,1],
               [0,0,0,0,0,0,0,0],
               [0,0,0,0,0,0,0,0],
               [2,0,2,0,2,0,2,0],
               [0,2,0,2,0,2,0,2],
               [2,0,2,0,2,0,2,0],]
        return board
    
    def getWinner(self,board=0):
        if board == 0 :
            board = self.board

        black, red = 0, 0 

        for i in range(0,ROWS):
            for j in range(0,COLS):
                if board[i][j] == 1: 
                    black += 1
                elif board[i][j] == 2: 
                    red += 1 

        if black == 0 :
            return 2
        if red == 0 :
            return 1
        
        black_successors = self.getSuccessorsBlack()
        red_successors = self.getSuccessorsRed()

        if not black_successors:
            return 2
        if not red_successors:
            return 1
        
        if self.numMoves == 1000:
            return 0
        return 0 
    
    def isDone(self, board=0):
        if board == 0:
            board = self.board

        if self.getWinner(board) == 0:
            return False
        else:
            return True 
        
    def getFeatures(self, board = 0):
        if board == 0:
            board = self.board

        # x1 = no of black pieces
        # x2 = no of red pieces
        # x3 = no of black king
        # x4 = no of red king
        # x5 = no of black pieces threatened 
        # x6 = no of red pieces threatened
            
        x1, x2, x3, x4, x5, x6 = 0, 0, 0, 0, 0, 0

        for i in range(0,ROWS):
            for j in range(0, COLS):
                if board[i][j] == 1:
                    x1 += 1
                elif board[i][j] == 2:
                    x2 += 1
                elif board[i][j] == -1:
                    x3 += 1
                    x1 += 1
                elif board[i][j] == -2:
                    x4 += 1 
                    x2 += 1

        black_threatened_pieces = set()

        for i in range(0, ROWS):
            for j in range(0, COLS):
                # Search red
                if board[i][j] == 2: # Red non king
                    
                    for k in range(0, 2):
                        threat_row = i + red_dx[k]
                        threat_col = j + red_dy[k]

                        if 0 <= threat_row < ROWS and 0 <= threat_col < COLS: # Check bounds
                            if board[threat_row][threat_col] == 1 or board[threat_row][threat_col] == -1: # Black piece
                                if 0 <= threat_row + red_dx[k] < ROWS and 0 <= threat_col + red_dy[k] < COLS and board[threat_row + red_dx[k]][threat_col + red_dy[k]] == 0: # Check bounds
                                    black_threatened_pieces.add((threat_row, threat_col))

                elif board[i][j] == -2: # red king
                    
                    for k in range(0, 4):
                        threat_row = i + king_dx[k]
                        threat_col = j + king_dy[k]

                        if 0 <= threat_row < ROWS and 0 <= threat_col < COLS: # Check bounds
                            if board[threat_row][threat_col] == 1 or board[threat_row][threat_col] == -1: # Black piece
                                if 0 <= threat_row + king_dx[k] < ROWS and 0 <= threat_col + king_dy[k] < COLS and board[threat_row + king_dx[k]][threat_col + king_dy[k]] == 0: # Check bounds
                                    black_threatened_pieces.add((threat_row, threat_col))

        x5 = len(black_threatened_pieces)


        red_threatened_pieces = set()

        for i in range(0, ROWS):
            for j in range(0, COLS):
                # Search black
                if board[i][j] == 1: # Black non king
                    
                    for k in range(0, 2):
                        threat_row = i + black_dx[k]
                        threat_col = j + black_dy[k]

                        if 0 <= threat_row < ROWS and 0 <= threat_col < COLS: # Check bounds
                            if board[threat_row][threat_col] == 2 or board[threat_row][threat_col] == -2: # red piece
                                if 0 <= threat_row + black_dx[k] < ROWS and 0 <= threat_col + black_dy[k] < COLS and board[threat_row + black_dx[k]][threat_col + black_dy[k]] == 0: # Check bounds
                                    red_threatened_pieces.add((threat_row, threat_col))

                elif board[i][j] == -1: # black king
                    
                    for k in range(0, 4):
                        threat_row = i + king_dx[k]
                        threat_col = j + king_dy[k]

                        if 0 <= threat_row < ROWS and 0 <= threat_col < COLS: # Check bounds
                            if board[threat_row][threat_col] == 2 or board[threat_row][threat_col] == -2: # red piece
                                if 0 <= threat_row + king_dx[k] < ROWS and 0 <= threat_col + king_dy[k] < COLS and board[threat_row + king_dx[k]][threat_col + king_dy[k]] == 0: # Check bounds
                                    red_threatened_pieces.add((threat_row, threat_col))

        x6 = len(red_threatened_pieces)

        return x1, x2, x3, x4, x5, x6 
    
    def getValidMoves(self, x, y, board=0):
        if board == 0:
            board = self.board

        valid_moves = []

        if board[x][y] == 1: # Black non king
            for k in range(0, 2):
                new_x = x + black_dx[k]
                new_y = y + black_dy[k]

                if 0 <= new_x < ROWS and 0 <= new_y < COLS: # Check bounds
                    if board[new_x][new_y] == 0: # If empty
                        valid_moves.append((new_x, new_y))
                    elif board[new_x][new_y] == 2 or board[new_x][new_y] == -2: # Red piece is there
                        if 0 <= new_x + black_dx[k] < ROWS and 0 <= new_y + black_dy[k] < COLS and board[new_x + black_dx[k]][new_y + black_dy[k]] == 0: # Check bounds
                            valid_moves.append((new_x + black_dx[k], new_y + black_dy[k]))

        elif board[x][y] == -1: # Black king
            for k in range(0, 4):
                new_x = x + king_dx[k]
                new_y = y + king_dy[k]

                if 0 <= new_x < ROWS and 0 <= new_y < COLS: # Check bounds
                    if board[new_x][new_y] == 0: # If empty
                        valid_moves.append((new_x, new_y))
                    elif board[new_x][new_y] == 2 or board[new_x][new_y] == -2: # Red piece is there
                        if 0 <= new_x + king_dx[k] < ROWS and 0 <= new_y + king_dy[k] < COLS and board[new_x + king_dx[k]][new_y + king_dy[k]] == 0: # Check bounds
                            valid_moves.append((new_x + king_dx[k], new_y + king_dy[k]))

        elif board[x][y] == 2: # Red non king
            for k in range(0, 2):
                new_x = x + red_dx[k]
                new_y = y + red_dy[k]

                if 0 <= new_x < ROWS and 0 <= new_y < COLS: # Check bounds
                    if board[new_x][new_y] == 0: # If empty
                        valid_moves.append((new_x, new_y))
                    elif board[new_x][new_y] == 1 or board[new_x][new_y] == -1: # Black piece is there
                        if 0 <= new_x + red_dx[k] < ROWS and 0 <= new_y + red_dy[k] < COLS and board[new_x + red_dx[k]][new_y + red_dy[k]] == 0: # Check bounds
                            valid_moves.append((new_x + red_dx[k], new_y + red_dy[k]))
        elif board[x][y] == -2: # Red king
            for k in range(0, 4):
                new_x = x + king_dx[k]
                new_y = y + king_dy[k]

                if 0 <= new_x < ROWS and 0 <= new_y < COLS: # Check bounds
                    if board[new_x][new_y] == 0: # If empty
                        valid_moves.append((new_x, new_y))
                    elif board[new_x][new_y] == 1 or board[new_x][new_y] == -1: # Black piece is there
                        if 0 <= new_x + king_dx[k] < ROWS and 0 <= new_y + king_dy[k] < COLS and board[new_x + king_dx[k]][new_y + king_dy[k]] == 0: # Check bounds
                            valid_moves.append((new_x + king_dx[k], new_y + king_dy[k]))

        return valid_moves
    
    def checkPromotion(self, board=0):
        if board==0:
            board = self.board

        for k in range(0, COLS):
            if board[ROWS - 1][k] == 1:
                board[ROWS - 1][k] = -1
            if board[0][k] == 2:
                board[0][k] = -2

    def getSuccessorsBlack(self):
        successors = []

        for i in range(0, ROWS):
            for j in range(0, COLS):

                if self.board[i][j] == 1 or self.board[i][j] == -1: # black piece 
                    valid_moves = self.getValidMoves(i, j)
                    for move in valid_moves:
                        new_i = move[0]
                        new_j = move[1]
                        if abs(new_i - i) == 2 and abs(new_j - j) == 2: # Hop
                            successor = copy.deepcopy(self.board)
                            successor[new_i][new_j] = self.board[i][j]
                            self.checkPromotion(successor)
                            successor[i][j] = 0
                            if new_i - i < 0 and new_j - j > 0:
                                successor[i - 1][j + 1] = 0
                            elif new_i - i > 0 and new_j - j > 0:
                                successor[i + 1][j + 1] = 0
                            elif new_i - i < 0 and new_j - j < 0:
                                successor[i - 1][j - 1] = 0
                            elif new_i - i > 0 and new_j - j < 0:
                                successor[i + 1][j - 1] = 0
                            successors.append(successor)
                        else: # Not hop
                            successor = copy.deepcopy(self.board)
                            successor[new_i][new_j] = self.board[i][j]
                            self.checkPromotion(successor)
                            successor[i][j] = 0
                            successors.append(successor)


        return successors
    
    def getSuccessorsRed(self):
        successors = []

        for i in range(0, ROWS):
            for j in range(0, COLS):

                if self.board[i][j] == 2 or self.board[i][j] == -2: # Red piece 
                    valid_moves = self.getValidMoves(i, j)
                    for move in valid_moves:
                        new_i = move[0]
                        new_j = move[1]
                        if abs(new_i - i) == 2 and abs(new_j - j) == 2: # Hop
                            successor = copy.deepcopy(self.board)
                            successor[new_i][new_j] = self.board[i][j]
                            self.checkPromotion(successor)
                            successor[i][j] = 0
                            if new_i - i < 0 and new_j - j > 0:
                                successor[i - 1][j + 1] = 0
                            elif new_i - i > 0 and new_j - j > 0:
                                successor[i + 1][j + 1] = 0
                            elif new_i - i < 0 and new_j - j < 0:
                                successor[i - 1][j - 1] = 0
                            elif new_i - i > 0 and new_j - j < 0:
                                successor[i + 1][j - 1] = 0
                            successors.append(successor)
                        else: # Not hop
                            successor = copy.deepcopy(self.board)
                            successor[new_i][new_j] = self.board[i][j]
                            self.checkPromotion(successor)
                            successor[i][j] = 0
                            successors.append(successor)


        return successors
    
    def printBoard(self, board=0):
        if board == 0:
            board = self.board
        
        for i in range(0, ROWS):
            for j in range(0, COLS):
                if board[i][j] == 1: # Black non king
                    print("(B )", end='')
                elif board[i][j] == -1: # Black king
                    print("(BK)", end='')
                elif board[i][j] == 2: # Red non king
                    print("(R )", end='')
                elif board[i][j] == -2: # Red king
                    print("(RK)", end='')
                else:
                    print("(  )", end='')
            print()

    def getHistory(self):
        return self.history
    
class PerformanceSystem:
    def __init__(self,board,hypothesis,mode = 1):
        self.board = board
        self.hypothesis = hypothesis
        self.mode = mode
        self.history = []        
        self.updateConstant = .001

    def setUpdateConstant(self, constant):
        self.updateConstant = constant

    def evaluateBoard(self,board):
        x1,x2,x3,x4,x5,x6 = self.board.getFeatures(board)

        w0,w1,w2,w3,w4,w5,w6 = self.hypothesis

        return w0 + w1*x1 + w2*x2 + w3*x3 + w4*x4 + w5*x5 + w6*x6

    def setBoard(self, board):
        self.board = board

    def getBoard(self):
        return self.board

    def setHypothesis(self, hypothesis):
        self.hypothesis = hypothesis

    def getHypothesis(self):
        return self.hypothesis

    def chooseRandom(self):
        if self.mode == 1:
            successors = self.board.getSuccessorsBlack()
        else:
            successors = self.board.getSuccessorsRed()
            
        randomBoard = successors[random.randint(0,len(successors)-1)]
        self.board.setBoard(randomBoard)

    def chooseMove(self):
        if self.mode == 1:
            successors = self.board.getSuccessorsBlack()
        else:
            successors = self.board.getSuccessorsRed()

        bestSuccessor = successors[0]
        bestValue = self.evaluateBoard(bestSuccessor)

        for successor in successors:
            value = self.evaluateBoard(successor)
            if value > bestValue:
                bestValue = value
                bestSuccessor = successor

        self.board.setBoard(bestSuccessor)


    def updateWeights(self,history,trainingExamples):
        for i in range(0,len(history)):
            w0,w1,w2,w3,w4,w5,w6 = self.hypothesis
            vEst = self.evaluateBoard(history[i])
            x1,x2,x3,x4,x5,x6 = trainingExamples[i][0]
            vTrain = trainingExamples[i][1]            

            w0 = w0 + self.updateConstant*(vTrain - vEst)
            w1 = w1 + self.updateConstant*(vTrain - vEst)*x1
            w2 = w2 + self.updateConstant*(vTrain - vEst)*x2
            w3 = w3 + self.updateConstant*(vTrain - vEst)*x3
            w4 = w4 + self.updateConstant*(vTrain - vEst)*x4
            w5 = w5 + self.updateConstant*(vTrain - vEst)*x5
            w6 = w6 + self.updateConstant*(vTrain - vEst)*x6

            self.hypothesis = w0,w1,w2,w3,w4,w5,w6

class Critic:
    def __init__(self,hypothesis,mode = 1):
        self.hypothesis = hypothesis
        self.mode = mode
        self.checker = ExperimentGenerator()
        
    def evaluateBoard(self,board):
        x1,x2,x3,x4,x5,x6 = self.checker.getFeatures(board)

        w0,w1,w2,w3,w4,w5,w6 = self.hypothesis

        return w0 + w1*x1 + w2*x2 + w3*x3 + w4*x4 + w5*x5 + w6*x6

    def setHypothesis(self,hypothesis):
        self.hypothesis = hypothesis

    def setMode(self,mode):
        self.mode = mode

    def getTrainingExamples(self,history):
        trainingExamples = []

        for i in range(0,len(history)):
            if(self.checker.isDone(history[i])):
                if(self.checker.getWinner(history[i]) == self.mode):
                    trainingExamples.append([self.checker.getFeatures(history[i]), 100])
                elif(self.checker.getWinner(history[i]) == 0):
                    trainingExamples.append([self.checker.getFeatures(history[i]), 0])
                else:
                    trainingExamples.append([self.checker.getFeatures(history[i]), -100])
            else:
                if i+2 >= len(history):
                    if(self.checker.getWinner(history[len(history)-1]) == 0):
                        trainingExamples.append([self.checker.getFeatures(history[i]), 0])
                    else:
                        trainingExamples.append([self.checker.getFeatures(history[i]), -100])
                else:
                    trainingExamples.append([self.checker.getFeatures(history[i]), self.evaluateBoard(history[i+2])])

        return trainingExamples
    
def draw_board(board, selected_piece=None, valid_moves=None):
    # Draw all squares first
    for row in range(8):
        for col in range(8):
            color = WHITE if (row + col) % 2 == 0 else BLUE
            pygame.draw.rect(screen, color, (col * GRID_SIZE, row * GRID_SIZE, GRID_SIZE, GRID_SIZE))

    # Draw all pieces next, if piece selected, draw it's valid positions
    for row in range(8):
        for col in range(8):
            piece = board[row][col]
            if piece == 1:
                pygame.draw.circle(screen, BLACK, (col * GRID_SIZE + GRID_SIZE // 2, row * GRID_SIZE + GRID_SIZE // 2), GRID_SIZE // 2 - 5)
             
            elif piece == 2:
                pygame.draw.circle(screen, RED, (col * GRID_SIZE + GRID_SIZE // 2, row * GRID_SIZE + GRID_SIZE // 2), GRID_SIZE // 2 - 5)
       
            if piece == 3:
                pygame.draw.circle(screen, BLACK, (col * GRID_SIZE + GRID_SIZE // 2, row * GRID_SIZE + GRID_SIZE // 2), GRID_SIZE // 2 - 5)
                draw_crown(col, row)
         
            elif piece == 4:
                pygame.draw.circle(screen, RED, (col * GRID_SIZE + GRID_SIZE // 2, row * GRID_SIZE + GRID_SIZE // 2), GRID_SIZE // 2 - 5)
                draw_crown(col, row)
                

def draw_crown(col, row):
    crown_rect = CROWN.get_rect(center=(col * GRID_SIZE + GRID_SIZE // 2, row * GRID_SIZE + GRID_SIZE // 2))
    screen.blit(CROWN, crown_rect)



pygame.init()

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Checkers")

board = ExperimentGenerator()
hypothesis1 = (.5,.5,.5,.5,.5,.5,.5)
hypothesis2 = (.5,.5,.5,.5,.5,.5,.5)
player1 = PerformanceSystem(board, hypothesis1, 1)
player2 = PerformanceSystem(board, hypothesis2, 2)
critic1 = Critic(hypothesis1, 1)
critic2 = Critic(hypothesis2, 2)

black_win = 0
red_win = 0
draws = 0

for i in range(0, 150):
    board = ExperimentGenerator()
    player1.setBoard(board)
    player2.setBoard(board)

    print(f"Game {i + 1}")
    while(not board.isDone()):
        if board.numMoves > 500:
            player1.chooseRandom()
        else:
            player1.chooseMove()
        draw_board(board.board)
        if board.isDone():
            break
        player2.chooseRandom()
        board.numMoves += 1
        draw_board(board.board)
        if board.numMoves == 1000:
            break
    
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        
        pygame.display.flip()

        pygame.time.Clock().tick(60)
    board.printBoard()
    print("FEATURES")
    x1, x2, x3, x4, x5, x6 = board.getFeatures()
    print(x1, x2, x3, x4, x5, x6)
    print("WEIGHTS")
    w0, w1, w2, w3, w4, w5, w6 = player1.getHypothesis()
    print("BLACK: ", w0, w1, w2, w3, w4, w5, w6)
    w0, w1, w2, w3, w4, w5, w6 = player2.getHypothesis()
    print("RED: ", w0, w1, w2, w3, w4, w5, w6)


    winner = board.getWinner()
        
    if(winner == 1):
        print ("BLACK WINS")
        black_win += 1
    elif(winner == 2):
        print ("RED WINS")
        red_win += 1
    elif(winner == 0):
        print ("DRAW GAME")
        draws += 1
    print("#################################################################")

    critic1.setHypothesis(player1.getHypothesis())
    critic2.setHypothesis(player2.getHypothesis())

    player1.updateWeights(board.getHistory(),critic1.getTrainingExamples(board.getHistory()))
    player2.updateWeights(board.getHistory(),critic2.getTrainingExamples(board.getHistory()))

print("Black wins: ", black_win)
print("Red wins: ", red_win)
print("Draws: ", draws)