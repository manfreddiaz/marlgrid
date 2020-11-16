from ..agents import GridAgentInterface
from ..base import MultiGridEnv, MultiGrid
from ..objects import *

VON_NEUMANN_DIR = np.array([
    [1, 0],
    [0, 1],
    [1, 1],
    [-1, 0],
    [0, -1],
    [-1, 1],
    [1, -1],
    [-1, -1]
])
GRAB_FLAG_REWARD = 100.
KEEPING_FLAG_REWARD = 10.
SCORING_REWARD = 1000.
RESPAWN_TIME = 8


class CTFAgentInterface(GridAgentInterface):

    def render(self, img):
        if self.active:
            tri_fn = point_in_triangle((0.12, 0.19), (0.87, 0.50), (0.12, 0.81), )
            tri_fn = rotate_fn(tri_fn, cx=0.5, cy=0.5, theta=0.5 * np.pi * (self.dir))
            fill_coords(img, tri_fn, COLORS[self.color])

    def render_post(self, tile):
        if not self.active:
            return tile

        if self.carrying is not None:
            c = COLORS[self.carrying.color]
            # Vertical quad
            post = point_in_rect(0.20, 0.25, 0.20, 0.45)
            post = rotate_fn(post, cx=0.5, cy=0.5, theta=0.5 * np.pi * (self.dir))
            fill_coords(tile, post, c)

            flag = point_in_triangle(
                (0.20, 0.20),
                (0.20, 0.10),
                (0.40, 0.15),
            )
            flag = rotate_fn(flag, cx=0.5, cy=0.5, theta=0.5 * np.pi * (self.dir))
            fill_coords(tile, flag , c)

        return tile



class TeamBase(WorldObj):
    def __init__(self, color='red'):
        super(TeamBase, self).__init__(color)

    def can_overlap(self):
        return True

    def render(self, img):
        c = COLORS[self.color]
        # Vertical quad

        fill_coords(img, point_in_rect(0.20, 0.80, 0.90, 0.95), c)

    def can_pickup(self):
        return False


class Flag(WorldObj):

    def __init__(self, color='red'):
        super(Flag, self).__init__(color)

    def can_overlap(self):
        return True

    def render(self, img):
        c = COLORS[self.color]
        # Vertical quad
        fill_coords(img, point_in_rect(0.35, 0.45, 0.31, 0.88), c)
        fill_coords(img, point_in_triangle(
            (0.35, 0.31),
            (0.80, 0.50),
            (0.35, 0.60),
        ), c)

    def can_pickup(self):
        return True


class CapturingFlagEnv(MultiGridEnv):
    mission = "get to the green square"
    metadata = {}

    def __init__(
            self,
            teams,
            agents_per_team,
            scores_to_win,
            grid_size=None,
            width=None,
            height=None,
            max_steps=100,
            reward_decay=True,
            seed=1337,
            respawn=False,
            ghost_mode=True,
            agent_spawn_kwargs={}
    ):
        self.num_teams = teams
        self.num_agents_per_team = agents_per_team
        self.scores_to_win = scores_to_win
        self.score = np.zeros(shape=teams)

        agents, self.teams_colors = self._create_teams()

        super().__init__(
            agents,
            grid_size,
            width,
            height,
            max_steps,
            reward_decay,
            seed,
            respawn,
            ghost_mode,
            agent_spawn_kwargs
        )

    def reset(self, **kwargs):
        for agent in self.agents:
            agent.agents = []
            agent.reset(new_episode=True)

        self._gen_grid(self.width, self.height)

        for agent in self.agents:
            if agent.spawn_delay == 0:
                # self.place_obj(agent, **self.agent_spawn_kwargs)
                agent.activate()

        self.step_count = 0
        obs = self.gen_obs()
        return obs

    def _gen_grid(self, width, height):
        self.grid = MultiGrid((width, height))
        self.grid.wall_rect(0, 0, width, height)

        for team in range(self.num_teams):
            # team base
            team_flag = Flag(color=self.teams_colors[team])
            self.put_obj(
                team_flag,
                self.np_random.randint(1, width - 2),
                self.np_random.randint(1, width - 2)
            )
            team_base = TeamBase(color=self.teams_colors[team])
            self.try_place_obj(team_base, team_flag.pos_init)
            # team members
            agent_low_index = self.num_agents_per_team * team
            agent_high_index = self.num_agents_per_team * team + self.num_agents_per_team
            for in_team_index, global_index in enumerate(range(agent_low_index, agent_high_index)):
                agent = self.agents[global_index]
                pos = self.place_obj(agent, team_flag.pos_init + VON_NEUMANN_DIR[in_team_index])
                agent.pos_init = pos

    def _create_teams(self):
        agents = []
        colors = list(sorted(COLORS.keys()))

        for index, team in enumerate(range(self.num_teams)):
            for agent in range(self.num_agents_per_team):
                agents.append(
                    CTFAgentInterface(
                        color=colors[index],
                        observation_style='rich'
                    )
                )
        return agents, colors[:self.num_teams]

    def _agent_step(self, agent_no, agent, action):
        cur_pos = agent.pos[:]
        cur_cell = self.grid.get(*cur_pos)
        fwd_pos = agent.front_pos[:]
        fwd_cell = self.grid.get(*fwd_pos)
        agent_moved = False

        # Rotate left
        if action == agent.actions.left:
            agent.dir = (agent.dir - 1) % 4

        # Rotate right
        elif action == agent.actions.right:
            agent.dir = (agent.dir + 1) % 4

        # Move forward
        elif action == agent.actions.forward:
            # Under the follow conditions, the agent can move forward.
            can_move = fwd_cell is None or fwd_cell.can_overlap()
            if self.ghost_mode is False and isinstance(fwd_cell, GridAgent):
                can_move = False

            if can_move:
                agent_moved = True
                # Add agent to new cell
                if fwd_cell is None:
                    self.grid.set(*fwd_pos, agent)
                    agent.pos = fwd_pos
                else:
                    fwd_cell.agents.append(agent)
                    agent.pos = fwd_pos

                # Remove agent from old cell
                if cur_cell == agent:
                    self.grid.set(*cur_pos, None)
                else:
                    assert cur_cell.can_overlap()
                    if agent in cur_cell.agents:  # TODO: This may be a potential bug on the original code.
                        cur_cell.agents.remove(agent)

                        # Add agent's agents to old cell
                for left_behind in agent.agents:
                    cur_obj = self.grid.get(*cur_pos)
                    if cur_obj is None:
                        self.grid.set(*cur_pos, left_behind)
                    elif cur_obj.can_overlap():
                        cur_obj.agents.append(left_behind)
                    else:  # How was "agent" there in teh first place?
                        raise ValueError("?!?!?!")

                # After moving, the agent shouldn't contain any other agents.
                agent.agents = []
                # test_integrity(f"After moving {agent.color} fellow")

                # Rewards can be got iff. fwd_cell has a "get_reward" method
                if hasattr(fwd_cell, 'get_reward'):
                    rwd = fwd_cell.get_reward(agent)
                    if bool(self.reward_decay):
                        rwd *= (1.0 - 0.9 * (self.step_count / self.max_steps))
                    step_rewards[agent_no] += rwd
                    agent.reward(rwd)

                if isinstance(fwd_cell, (Lava, Goal)):
                    agent.done = True

        elif action == agent.actions.pickup:
            if fwd_cell and fwd_cell.can_pickup():
                if agent.carrying is None:
                    if agent.color != fwd_cell.color:  # can't take its own flag
                        agent.carrying = fwd_cell
                        agent.carrying.cur_pos = np.array([-1, -1])
                    else:
                        self.grid.set(*fwd_cell.pos_init, fwd_cell)  # return flag to base.

                self.grid.set(*fwd_pos, None)
            else:
                pass

        # Drop an object
        elif action == agent.actions.drop:
            if agent.carrying:
                if fwd_cell and fwd_cell.color == agent.color:
                    print('score')  # TODO: ADD  scoring reward
                    self.grid.set(*agent.carrying.pos_init, agent.carrying)
                    agent.carrying.cur_pos = agent.carrying.pos_init
                    agent.carrying = None
                    self.score[agent_no // self.num_teams] += 1
                elif not fwd_cell:
                    self.grid.set(*fwd_pos, agent.carrying)
                    agent.carrying.cur_pos = fwd_pos
                    agent.carrying = None
            else:
                pass

        # Toggle/activate an, for agents this is the equivalent to tag
        elif action == agent.actions.toggle:
            if fwd_cell:
                if "Agent" in fwd_cell.type:
                    fwd_cell.deactivate()
                    fwd_cell.spawn_delay = RESPAWN_TIME
                    self.grid.set(*fwd_pos, None)
            else:
                pass

        # Done action (not used by default)
        elif action == agent.actions.done:
            pass

        else:
            raise ValueError(f"Environment can't handle action {action}.")

        agent.on_step(fwd_cell if agent_moved else None)

    def step(self, actions):
        # Spawn agents if it's time.
        for agent in self.agents:
            if not agent.active and not agent.done and agent.spawn_delay == 0:
                self.try_place_obj(agent, agent.pos_init)
                agent.activate()
            elif agent.spawn_delay > 0:
                agent.spawn_delay -= 1

        assert len(actions) == len(self.agents)

        step_rewards = np.zeros((len(self.agents, )), dtype=np.float)

        self.step_count += 1

        iter_agents = list(enumerate(zip(self.agents, actions)))
        iter_order = np.arange(len(iter_agents))
        # TODO: decides who tag's who in simultaneous tagging
        self.np_random.shuffle(iter_order)
        for shuffled_ix in iter_order:
            agent_no, (agent, action) = iter_agents[shuffled_ix]
            agent.step_reward = 0

            if agent.active:
                self._agent_step(agent_no, agent, action)

        # the game is over if we reach the max_steps of if a team got the scores to win
        done = (self.step_count >= self.max_steps) or any(self.score[self.score == self.scores_to_win])
        print(self.step_count)
        obs = [self.gen_agent_obs(agent) for agent in self.agents]

        return obs, step_rewards, done, {}