import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from shapely.plotting import plot_polygon, plot_points

def plot_collision(ego_polygon: Polygon, agent_polygon: Polygon, t: int = 0):
    fig = plt.figure(1, figsize=(12,12), dpi=90)
    ax = fig.add_subplot()
    plot_polygon(ego_polygon, ax=ax, add_points=False, color='r', label="ego_polygon")
    plot_points(ego_polygon, ax=ax, color='r', alpha=0.7)
    plot_polygon(agent_polygon, ax=ax, add_points=False, color='b', label="agent_polygon")
    plot_points(agent_polygon, ax=ax, color='b', alpha=0.7)
    ax.set_aspect('equal', 'box')
    ax.set_title('collision detected at time step %d' %t)
    ax.legend(loc="upper left")
    plt.pause(0.0001)
    plt.show()
    return

def main():
    ego_polygon = Polygon([
                            (293.36682276178567, -332.93349763735245), 
                            (294.6254807615479, -331.3792196571847), 
                            (298.2003201159338, -334.27413305663794), 
                            (296.9416621161715, -335.8284110368057),
                            (293.36682276178567, -332.93349763735245)
                            ])
    agent_polygon = Polygon([
                            (291.74168431877877, -337.51043061752785),
                            (291.7652703621569, -335.1945749486635), 
                            (298.70265568122124, -335.26522938247217), 
                            (298.6790696378431, -337.5810850513365), 
                            (291.74168431877877, -337.51043061752785)
                            ])
    plot_collision(ego_polygon, agent_polygon)

if __name__ == '__main__':
    main()
