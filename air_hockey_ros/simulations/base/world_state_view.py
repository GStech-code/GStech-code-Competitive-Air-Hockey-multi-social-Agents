import pygame

WHITE_COLOR = (255, 255, 255)
BLACK_COLOR = (0, 0, 0)
LIGHT_GREY_COLOR = (240, 240, 240)
LIGHT_BLUE_COLOR = (128, 128, 255)
AZURE_COLOR = (0, 160, 255)
LIGHT_RED_COLOR = (255, 80, 80)
WARM_RED_COLOR = (255, 96, 56)

class PygameView:
    def __init__(self):
        pygame.init()
        self.clock = pygame.time.Clock()
        self._running = True

    def reset(self, **params):
        self.width = int(params.get("width", 800))
        self.height = int(params.get("height", 800))
        self.title = str(params.get("title", "Air Hockey"))
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption(self.title)
        self.hz = int(params.get("hz", 60))
        self.midx = self.width // 2
        self.cy = self.height // 2
        self.goal_gap_half = int(params.get("goal_gap_half", self.height // 5))
        self.puck_radius = int(params.get("puck_radius", 12))
        self.paddle_radius = int(params.get("paddle_radius", 20))
        num_agents_team_a = params.get("num_agents_team_a", 0)
        num_agents_team_b = params.get("num_agents_team_b", 0)
        self.colors = [AZURE_COLOR] * num_agents_team_a + [WARM_RED_COLOR] * num_agents_team_b

    def pump_events(self) -> bool:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                self._running = False
        return self._running

    def draw(self, world_state: dict, scores: dict) -> bool:
        if not self._running:
            return False
        if not self.pump_events():
            return False

        s = self.screen
        s.fill(BLACK_COLOR)

        # center line
        pygame.draw.line(s, WHITE_COLOR, (self.midx, 0), (self.midx, self.height), 2)

        # goals (vertical side gaps)
        # left posts
        pygame.draw.line(s, LIGHT_BLUE_COLOR, (0, 0), (0, self.cy - self.goal_gap_half), 4)
        pygame.draw.line(s, LIGHT_BLUE_COLOR, (0, self.cy + self.goal_gap_half), (0, self.height), 4)
        # right posts
        pygame.draw.line(s, LIGHT_RED_COLOR, (self.width - 1, 0), (self.width - 1, self.cy - self.goal_gap_half), 4)
        pygame.draw.line(s, LIGHT_RED_COLOR, (self.width - 1, self.cy + self.goal_gap_half),
                                            (self.width - 1, self.height), 4)

        # puck
        px, py = int(world_state["puck_x"]), int(world_state["puck_y"])
        pygame.draw.circle(s, LIGHT_GREY_COLOR, (px, py), self.puck_radius)

        # agents (A then B)
        ax, ay = world_state["agent_x"], world_state["agent_y"]
        for i in range(len(ax)):
            pygame.draw.circle(s, self.colors[i], (int(ax[i]), int(ay[i])), self.paddle_radius)

        # scores (simple bars)
        team_a_score = scores["team_a_score"]
        team_b_score = scores["team_b_score"]
        for k in range(min(10, team_a_score)):
            pygame.draw.rect(s, AZURE_COLOR, pygame.Rect(20 + k*18, 10, 14, 8))
        for k in range(min(10, team_b_score)):
            pygame.draw.rect(s, LIGHT_RED_COLOR, pygame.Rect(self.width - 20 - (k+1)*18, 10, 14, 8))

        pygame.display.flip()
        return True

    def tick(self):
        self.clock.tick(self.hz)

    def close(self):
        if self._running:
            self._running = False
            pygame.quit()
