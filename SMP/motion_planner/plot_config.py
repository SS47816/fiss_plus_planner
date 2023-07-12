class PlotConfig:
    DO_PLOT = False
    JUPYTER_NOTEBOOK = False
    PLOT_LEGEND = False
    PLOT_COLLISION_STEPS = False
    PLOTTING_PARAMS = {
        0: ('gold', 'solid', 2.0),
        1: ('gold', '--', 2.0),
        2: ('red', 'solid', 2.0),
        3: ('grey', 'solid', 2.0),
        4: ('green', 'solid', 3.0)
    }
    SAVE_FIG = False
    OUTPUT_FORMAT = 'svg'


class DefaultPlotConfig(PlotConfig):
    def __init__(self, DO_PLOT=True):
        super(DefaultPlotConfig, self).__init__()
        self.DO_PLOT = DO_PLOT
        self.SAVE_FIG = False

        self.JUPYTER_NOTEBOOK = True
        self.PLOT_LEGEND = True
        self.PLOT_COLLISION_STEPS = True


class StudentScriptPlotConfig(PlotConfig):
    def __init__(self, DO_PLOT=False):
        super(StudentScriptPlotConfig, self).__init__()
        self.DO_PLOT = DO_PLOT
        self.SAVE_FIG = False

        self.JUPYTER_NOTEBOOK = False
        self.PLOT_LEGEND = True
        self.PLOT_COLLISION_STEPS = True
