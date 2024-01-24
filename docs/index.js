importScripts("https://cdn.jsdelivr.net/pyodide/v0.24.1/full/pyodide.js");

function sendPatch(patch, buffers, msg_id) {
  self.postMessage({
    type: 'patch',
    patch: patch,
    buffers: buffers
  })
}

async function startApplication() {
  console.log("Loading pyodide!");
  self.postMessage({type: 'status', msg: 'Loading pyodide'})
  self.pyodide = await loadPyodide();
  self.pyodide.globals.set("sendPatch", sendPatch);
  console.log("Loaded!");
  await self.pyodide.loadPackage("micropip");
  const env_spec = ['https://cdn.holoviz.org/panel/wheels/bokeh-3.3.0-py3-none-any.whl', 'https://cdn.holoviz.org/panel/1.3.7/dist/wheels/panel-1.3.7-py3-none-any.whl', 'pyodide-http==0.2.1', 'numpy', 'pandas']
  for (const pkg of env_spec) {
    let pkg_name;
    if (pkg.endsWith('.whl')) {
      pkg_name = pkg.split('/').slice(-1)[0].split('-')[0]
    } else {
      pkg_name = pkg
    }
    self.postMessage({type: 'status', msg: `Installing ${pkg_name}`})
    try {
      await self.pyodide.runPythonAsync(`
        import micropip
        await micropip.install('${pkg}');
      `);
    } catch(e) {
      console.log(e)
      self.postMessage({
	type: 'status',
	msg: `Error while installing ${pkg_name}`
      });
    }
  }
  console.log("Packages loaded!");
  self.postMessage({type: 'status', msg: 'Executing code'})
  const code = `
  
import asyncio

from panel.io.pyodide import init_doc, write_doc

init_doc()

#!/usr/bin/env python

import pandas as pd
import numpy as np
from bokeh.plotting import figure
from bokeh.layouts import layout
from bokeh.models import (
    ColumnDataSource,
    Range1d,
    Slider,
    Button,
    TextInput,
    LabelSet,
    Circle,
    Div,
    Tabs,
    TabPanel,
    CustomJS,
)
import panel as pn

SIZING_MODE = "stretch_both"


class MATRIX_PROFILE:
    def __init__(self):
        self.sizing_mode = SIZING_MODE
        self.window = None
        self.m = None

        self.df = None
        self.ts_cds = None
        self.quad_cds = None
        self.pattern_match_cds = None
        self.dist_cds = None
        self.circle_cds = None

        self.ts_plot = None
        self.mp_plot = None
        self.pm_plot = None
        self.logo_div = None

        self.slider = None
        self.play_btn = None
        self.txt_inp = None
        self.pattern_btn = None
        self.match_btn = None
        self.reset_btn = None
        self.idx = None
        self.min_distance_idx = None

        self.animate_id = None

    def get_df_from_file(self):
        # raw_df = pd.read_csv('raw.csv')

        # mp_df = pd.read_csv('matrix_profile.csv')
        base_url = "https://raw.githubusercontent.com/seanlaw/stumpy-live-demo/master"
        raw_df = pd.read_csv(base_url + "/raw.csv")

        mp_df = pd.read_csv(base_url + "/matrix_profile.csv")

        self.window = raw_df.shape[0] - mp_df.shape[0] + 1
        self.m = raw_df.shape[0] - mp_df.shape[0] + 1
        self.min_distance_idx = mp_df["distance"].argmin()

        df = pd.merge(raw_df, mp_df, left_index=True, how="left", right_index=True)

        return df.reset_index()

    def get_ts_dict(self, df):
        return self.df.to_dict(orient="list")

    def get_circle_dict(self, df):
        return self.df[["index", "y"]].to_dict(orient="list")

    def get_quad_dict(self, df, pattern_idx=0, match_idx=None):
        if match_idx is None:
            match_idx = df.loc[pattern_idx, "idx"].astype(int)
        quad_dict = dict(
            pattern_left=[pattern_idx],
            pattern_right=[pattern_idx + self.window - 1],
            pattern_top=[max(df["y"])],
            pattern_bottom=[0],
            match_left=[match_idx],
            match_right=[match_idx + self.window - 1],
            match_top=[max(df["y"])],
            match_bottom=[0],
            vert_line_left=[pattern_idx - 5],
            vert_line_right=[pattern_idx + 5],
            vert_line_top=[max(df["distance"])],
            vert_line_bottom=[0],
            hori_line_left=[0],
            hori_line_right=[max(df["index"])],
            hori_line_top=[df.loc[pattern_idx, "distance"] - 0.01],
            hori_line_bottom=[df.loc[pattern_idx, "distance"] + 0.01],
        )
        return quad_dict

    def get_custom_quad_dict(self, df, pattern_idx=0, match_idx=None):
        if match_idx is None:
            match_idx = df.loc[pattern_idx, "idx"].astype(int)
        quad_dict = dict(
            pattern_left=[pattern_idx],
            pattern_right=[pattern_idx + self.window - 1],
            pattern_top=[max(df["y"])],
            pattern_bottom=[0],
            match_left=[match_idx],
            match_right=[match_idx + self.window - 1],
            match_top=[max(df["y"])],
            match_bottom=[0],
            vert_line_left=[match_idx - 5],
            vert_line_right=[match_idx + 5],
            vert_line_top=[max(df["distance"])],
            vert_line_bottom=[0],
            hori_line_left=[0],
            hori_line_right=[max(df["index"])],
            hori_line_top=[df.loc[match_idx, "distance"] - 0.01],
            hori_line_bottom=[df.loc[match_idx, "distance"] + 0.01],
        )
        return quad_dict

    def get_pattern_match_dict(self, df, pattern_idx=0, match_idx=None):
        if match_idx is None:
            match_idx = df["idx"].loc[pattern_idx].astype(int)
        pattern_match_dict = dict(
            index=list(range(self.window)),
            pattern=df["y"].loc[pattern_idx : pattern_idx + self.window - 1],
            match=df["y"].loc[match_idx : match_idx + self.window - 1],
        )

        return pattern_match_dict

    def get_ts_plot(self, color="black"):
        """
        Time Series Plot
        """
        ts_plot = figure(
            toolbar_location="above",
            sizing_mode=self.sizing_mode,
            title="Raw Time Series or Sequence",
            tools=["reset"],
        )
        q = ts_plot.quad(
            "pattern_left",
            "pattern_right",
            "pattern_top",
            "pattern_bottom",
            source=self.quad_cds,
            name="pattern_quad",
            color="#54b847",
        )
        q.visible = False
        q = ts_plot.quad(
            "match_left",
            "match_right",
            "match_top",
            "match_bottom",
            source=self.quad_cds,
            name="match_quad",
            color="#696969",
            alpha=0.5,
        )
        q.visible = False
        ts_plot.line(x="index", y="y", source=self.ts_cds, color=color)
        ts_plot.x_range = Range1d(
            0, max(self.df["index"]), bounds=(0, max(self.df["x"]))
        )
        ts_plot.y_range = Range1d(0, max(self.df["y"]), bounds=(0, max(self.df["y"])))

        c = ts_plot.circle(
            x="index", y="y", source=self.circle_cds, size=0, line_color="white"
        )
        c.selection_glyph = Circle(line_color="white")
        c.nonselection_glyph = Circle(line_color="white")

        return ts_plot

    def get_dist_dict(self, df, pattern_idx=0):
        dist = df["distance"]
        max_dist = dist.max()
        x_offset = self.df.shape[0] - self.window / 2
        y_offset = max_dist / 2
        distance = dist.loc[pattern_idx]
        text = distance.round(1).astype(str)
        gauge_dict = dict(x=[0 + x_offset], y=[0 + y_offset], text=[text])

        return gauge_dict

    def get_mp_plot(self):
        """
        Matrix Profile Plot
        """
        mp_plot = figure(
            x_range=self.ts_plot.x_range,
            toolbar_location=None,
            sizing_mode=self.sizing_mode,
            title="Matrix Profile (All Minimum Distances)",
        )
        q = mp_plot.quad(
            "vert_line_left",
            "vert_line_right",
            "vert_line_top",
            "vert_line_bottom",
            source=self.quad_cds,
            name="pattern_start",
            color="#54b847",
        )
        q.visible = False
        q = mp_plot.quad(
            "hori_line_left",
            "hori_line_right",
            "hori_line_top",
            "hori_line_bottom",
            source=self.quad_cds,
            name="match_dist",
            color="#696969",
            alpha=0.5,
        )
        q.visible = False
        mp_plot.line(x="index", y="distance", source=self.ts_cds, color="black")
        mp_plot.x_range = Range1d(
            0, self.df.shape[0] + 1, bounds=(0, self.df.shape[0] + 1)
        )
        mp_plot.y_range = Range1d(
            0, max(self.df["distance"]), bounds=(0, max(self.df["distance"]))
        )

        label = LabelSet(
            x="x",
            y="y",
            text="text",
            source=self.dist_cds,
            text_align="center",
            name="gauge_label",
            text_color="black",
            text_font_size="30pt",
        )
        mp_plot.add_layout(label)

        return mp_plot

    def get_pm_plot(self):
        """
        Pattern-Match Plot
        """
        pm_plot = figure(
            toolbar_location=None,
            sizing_mode=self.sizing_mode,
            title="Pattern Match Overlay",
        )
        l = pm_plot.line(
            "index",
            "pattern",
            source=self.pattern_match_cds,
            name="pattern_line",
            color="#54b847",
            line_width=2,
        )
        l.visible = False
        l = pm_plot.line(
            "index",
            "match",
            source=self.pattern_match_cds,
            name="match_line",
            color="#696969",
            alpha=0.5,
            line_width=2,
        )
        l.visible = False

        return pm_plot

    def get_logo_div(self):
        """
        STUMPY logo
        """
        text = "<a href='https://stumpy.readthedocs.io/en/latest/'>"
        text += "<img src='https://raw.githubusercontent.com/TDAmeritrade/stumpy/main"
        text += "/docs/images/stumpy_logo_small.png'></a>"
        logo_div = Div(text=text)

        return logo_div

    def get_slider(self, value=0):
        slider = Slider(
            start=0.0,
            end=max(self.df["index"]) - self.window,
            value=value,
            step=1,
            title="Subsequence",
            sizing_mode=self.sizing_mode,
        )
        return slider

    def get_play_button(self):
        play_btn = Button(label="► Play")
        play_btn.on_click(self.animate)
        return play_btn

    def get_text_input(self):
        txt_inp = TextInput(sizing_mode=self.sizing_mode)
        return txt_inp

    def get_buttons(self):
        pattern_btn = Button(label="Show Motif", sizing_mode=self.sizing_mode)
        match_btn = Button(label="Show Nearest Neighbor", sizing_mode=self.sizing_mode)
        reset_btn = Button(label="Reset", sizing_mode=self.sizing_mode)
        return pattern_btn, match_btn, reset_btn

    def update_plots(self, attr, new, old):
        self.quad_cds.data = self.get_quad_dict(self.df, self.slider.value)
        self.pattern_match_cds.data = self.get_pattern_match_dict(
            self.df, self.slider.value
        )
        self.dist_cds.data = self.get_dist_dict(self.df, self.slider.value)

    def custom_update_plots(self, attr, new, old):
        self.quad_cds.data = self.get_custom_quad_dict(
            self.df, self.pattern_idx, self.slider.value
        )
        self.pattern_match_cds.data = self.get_pattern_match_dict(
            self.df, self.pattern_idx, self.slider.value
        )
        self.dist_cds.data = self.get_dist_dict(self.df, self.slider.value)

    def show_hide_pattern(self):
        pattern_quad = self.ts_plot.select(name="pattern_quad")[0]
        pattern_start = self.mp_plot.select(name="pattern_start")[0]
        pattern_line = self.pm_plot.select(name="pattern_line")[0]
        if pattern_quad.visible:
            pattern_start.visible = False
            pattern_line.visible = False
            pattern_quad.visible = False
            self.pattern_btn.label = "Show Motif"
        else:
            pattern_start.visible = True
            pattern_line.visible = True
            pattern_quad.visible = True
            self.pattern_btn.label = "Hide Motif"

    def show_hide_match(self):
        match_quad = self.ts_plot.select(name="match_quad")[0]
        match_dist = self.mp_plot.select(name="match_dist")[0]
        match_line = self.pm_plot.select(name="match_line")[0]
        if match_quad.visible:
            match_dist.visible = False
            match_line.visible = False
            match_quad.visible = False
            self.match_btn.label = "Show Nearest Neighbor"
        else:
            match_dist.visible = True
            match_line.visible = True
            match_quad.visible = True
            self.match_btn.label = "Hide Nearest Neighbor"

    def update_slider(self, attr, old, new):
        self.slider.value = int(self.txt_inp.value)

    def animate(self):
        if self.play_btn.label == "► Play":
            self.play_btn.label = "❚❚ Pause"
            # self.animate_id = curdoc().add_periodic_callback(self.update_animate, 50)
            self.animate_id = pn.state.add_periodic_callback(self.update_animate, 50)
        else:
            self.play_btn.label = "► Play"
            # curdoc().remove_periodic_callback(self.animate_id)
            self.animate_id.stop()

    def update_animate(self, shift=50):
        if self.window < self.m:  # Probably using box select
            start = self.slider.value
            end = start + shift
            if self.df.loc[start:end, "distance"].min() <= 15:
                self.slider.value = self.df.loc[start:end, "distance"].idxmin()
                self.animate()
            elif self.slider.value + shift <= self.slider.end:
                self.slider.value = self.slider.value + shift
            else:
                self.slider.value = 0
        elif self.slider.value + shift <= self.slider.end:
            self.slider.value = self.slider.value + shift
        else:
            self.slider.value = 0

    def reset(self):
        self.sizing_mode = "stretch_both"
        self.window = self.m

        self.default_idx = self.min_distance_idx
        self.df = self.get_df_from_file()
        self.ts_cds.data = self.get_ts_dict(self.df)
        self.mp_plot.y_range.end = max(self.df["distance"])
        self.mp_plot.title.text = "Matrix Profile (All Minimum Distances)"
        self.mp_plot.y_range.bounds = (0, max(self.df["distance"]))
        self.quad_cds.data = self.get_quad_dict(self.df, pattern_idx=self.default_idx)
        self.pattern_match_cds.data = self.get_pattern_match_dict(
            self.df, pattern_idx=self.default_idx
        )
        self.dist_cds.data = self.get_dist_dict(self.df, pattern_idx=self.default_idx)
        self.circle_cds.data = self.get_circle_dict(self.df)
        # Remove callback and add old callback
        if self.custom_update_plots in self.slider._callbacks["value"]:
            self.slider.remove_on_change("value", self.custom_update_plots)
            self.slider.on_change("value", self.update_plots)
        self.slider.end = self.df.shape[0] - self.window
        self.slider.value = self.default_idx

    def get_data(self):
        self.df = self.get_df_from_file()
        self.default_idx = self.min_distance_idx
        self.ts_cds = ColumnDataSource(self.get_ts_dict(self.df))
        self.quad_cds = ColumnDataSource(
            self.get_quad_dict(self.df, pattern_idx=self.default_idx)
        )
        self.pattern_match_cds = ColumnDataSource(
            self.get_pattern_match_dict(self.df, pattern_idx=self.default_idx)
        )
        self.dist_cds = ColumnDataSource(
            self.get_dist_dict(self.df, pattern_idx=self.default_idx)
        )
        self.circle_cds = ColumnDataSource(self.get_circle_dict(self.df))

    def get_plots(self, ts_plot_color="black"):
        self.ts_plot = self.get_ts_plot(color=ts_plot_color)
        self.mp_plot = self.get_mp_plot()
        self.pm_plot = self.get_pm_plot()

    def get_widgets(self):
        self.slider = self.get_slider(value=self.default_idx)
        self.play_btn = self.get_play_button()
        self.txt_inp = self.get_text_input()
        self.pattern_btn, self.match_btn, self.reset_btn = self.get_buttons()
        self.logo_div = self.get_logo_div()

    def set_callbacks(self):
        self.slider.on_change("value", self.update_plots)
        self.pattern_btn.on_click(self.show_hide_pattern)
        self.match_btn.on_click(self.show_hide_match)
        self.reset_btn.on_click(self.reset)
        self.txt_inp.on_change("value", self.update_slider)

    def get_layout(self):
        self.get_data()
        self.get_plots()
        self.get_widgets()
        self.set_callbacks()

        l = layout(
            [
                [self.ts_plot],
                [self.mp_plot],
                [self.pm_plot],
                [self.slider],
                [self.pattern_btn, self.match_btn, self.play_btn, self.logo_div],
            ],
            sizing_mode=self.sizing_mode,
        )

        return l


class DISTANCE_PROFILE:
    def __init__(self):
        self.sizing_mode = SIZING_MODE
        self.window = None
        self.m = None

        self.df = None
        self.T = None
        self.pattern_idx_cds = None
        self.ts_cds = None
        self.hidden_ts_cds = None
        self.pattern_cds = None
        self.match_cds = None
        self.dp_cds = None

        self.reset_btn = None
        self.find_btn = None

        self.ts_plot = None
        self.dp_plot = None
        self.pm_plot = None

    def get_df_from_file(self):
        # raw_df = pd.read_csv('raw.csv')

        # mp_df = pd.read_csv('matrix_profile.csv')
        base_url = "https://raw.githubusercontent.com/seanlaw/stumpy-live-demo/master"
        raw_df = pd.read_csv(base_url + "/raw.csv")

        mp_df = pd.read_csv(base_url + "/matrix_profile.csv")

        self.window = raw_df.shape[0] - mp_df.shape[0] + 1
        self.m = raw_df.shape[0] - mp_df.shape[0] + 1
        self.min_distance_idx = mp_df["distance"].argmin()

        df = pd.merge(raw_df, mp_df, left_index=True, how="left", right_index=True)

        return df.reset_index()

    def get_ts_dict(self, df):
        return self.df.to_dict(orient="list")

    def get_ts_plot(self, color="black"):
        """
        Time Series Plot
        """
        ts_plot = figure(
            toolbar_location=None,
            sizing_mode=self.sizing_mode,
            title="Raw Time Series or Sequence",
            tools=["box_select"],
        )
        ts_plot.scatter(x="index", y="y", source=self.hidden_ts_cds, color="white")
        ts_plot.line(x="index", y="y", source=self.ts_cds, color=color)
        ts_plot.x_range = Range1d(
            0, max(self.df["index"]), bounds=(0, max(self.df["x"]))
        )
        ts_plot.y_range = Range1d(0, max(self.df["y"]), bounds=(0, max(self.df["y"])))

        return ts_plot

    def get_dp_plot(self, color="black"):
        dp_plot = figure(
            x_range=self.ts_plot.x_range,
            toolbar_location=None,
            sizing_mode=self.sizing_mode,
            title="Distance Profile",
        )
        dp_plot.y_range.start = 0
        dp_plot.line(x="index", y="y", source=self.dp_cds, color=color)

        return dp_plot

    def get_pm_plot(self):
        pm_plot = figure(
            toolbar_location=None,
            sizing_mode=self.sizing_mode,
            title="Pattern Match Overlay",
        )
        pm_plot.line(
            x="index", y="y", source=self.pattern_cds, color="#54b847", line_width=2
        )
        pm_plot.line(
            x="index",
            y="y",
            source=self.match_cds,
            color="#696969",
            line_width=2,
            alpha=0.5,
        )

        return pm_plot

    def get_data(self):
        self.df = self.get_df_from_file()
        self.ts_cds = ColumnDataSource(self.get_ts_dict(self.df))
        self.T = np.array(self.ts_cds.data.get("y"))
        self.pattern_idx_cds = ColumnDataSource(data=dict(index=[]))
        self.hidden_ts_cds = ColumnDataSource(self.get_ts_dict(self.df))
        self.pattern_cds = ColumnDataSource(data=dict(index=[], y=[]))
        self.match_cds = ColumnDataSource(data=dict(index=[], y=[]))
        self.dp_cds = ColumnDataSource(data=dict(index=[], y=[]))

    def get_plots(self, ts_plot_color="black"):
        self.ts_plot = self.get_ts_plot(color=ts_plot_color)
        self.dp_plot = self.get_dp_plot()
        self.pm_plot = self.get_pm_plot()

    def update_distance_profile_match(self):
        Q = np.array(self.pattern_cds.data.get("y"))
        μ_Q = np.mean(Q)
        σ_Q = np.std(Q)
        m = len(Q)
        D = np.empty(len(self.T) - m + 1)
        for i in range(len(D)):
            QT = np.dot(Q, self.T[i : i + m])
            M_T = np.mean(self.T[i : i + m])
            Σ_T = np.std(self.T[i : i + m])
            denom = denom = (σ_Q * Σ_T) * m
            ρ = (QT - (μ_Q * M_T) * m) / denom
            D[i] = np.sqrt(np.abs(2 * m * (1.0 - ρ)))

        pattern_idx = self.pattern_idx_cds.data.get("index")[0]
        excl_zone = int(np.ceil(m / 4))
        zone_start = max(0, pattern_idx - excl_zone)
        zone_stop = min(D.shape[-1], pattern_idx + excl_zone)
        D[..., zone_start : zone_stop + 1] = np.nan

        self.dp_cds.data = dict(index=list(range(len(D))), y=list(D))
        match_idx = np.nanargmin(D)
        self.match_cds.data = dict(
            index=list(range(len(Q))),
            y=list(self.T[match_idx : match_idx + m]),
        )

    def clear_plots(self, attr, old, new):
        self.match_cds.data = dict(index=[], y=[])
        self.dp_cds.data = dict(index=[], y=[])

    def reset_plots(self):
        self.pattern_cds.data = dict(index=[], y=[])
        self.match_cds.data = dict(index=[], y=[])
        self.dp_cds.data = dict(index=[], y=[])

    def get_buttons(self):
        self.reset_btn = Button(label="Reset")
        self.find_btn = Button(label="Find Pattern")

    def set_callbacks(self):
        self.hidden_ts_cds.selected.on_change("indices", self.clear_plots)
        self.hidden_ts_cds.selected.js_on_change(
            "indices",
            CustomJS(
                args=dict(
                    hidden_ts_cds=self.hidden_ts_cds,
                    pattern_cds=self.pattern_cds,
                    pattern_idx_cds=self.pattern_idx_cds,
                ),
                code="""
                var inds = cb_obj.indices;
                var hidden_ts_cds_data = hidden_ts_cds.data;
                var selected = {'index': [], 'y': []};
                var pattern_idx = {'index': []};
                pattern_idx['index'].push(inds[0]);
                for (var i = 0; i <= inds[inds.length-1] - inds[0] + 1; i++) {
                    selected['index'].push(i);
                    selected['y'].push(hidden_ts_cds_data['y'][inds[0] + i]);
                }
                pattern_cds.data = selected;
                pattern_idx_cds.data = pattern_idx;
                """,
            ),
        )
        self.find_btn.on_click(self.update_distance_profile_match)
        self.reset_btn.on_click(self.reset_plots)

    def get_layout(self):
        self.get_data()
        self.get_plots()
        self.get_buttons()
        self.set_callbacks()

        l = layout(
            [
                [self.ts_plot],
                [self.dp_plot],
                [self.pm_plot],
                [self.find_btn, self.reset_btn],
            ],
            sizing_mode=self.sizing_mode,
        )

        return l


mp = MATRIX_PROFILE()
mp_layout = TabPanel(child=mp.get_layout(), title="Matrix Profile")
dp = DISTANCE_PROFILE()
dp_layout = TabPanel(child=dp.get_layout(), title="Distance Profile")

pn.state.curdoc.add_root(Tabs(tabs=[mp_layout, dp_layout], sizing_mode=SIZING_MODE))


await write_doc()
  `

  try {
    const [docs_json, render_items, root_ids] = await self.pyodide.runPythonAsync(code)
    self.postMessage({
      type: 'render',
      docs_json: docs_json,
      render_items: render_items,
      root_ids: root_ids
    })
  } catch(e) {
    const traceback = `${e}`
    const tblines = traceback.split('\n')
    self.postMessage({
      type: 'status',
      msg: tblines[tblines.length-2]
    });
    throw e
  }
}

self.onmessage = async (event) => {
  const msg = event.data
  if (msg.type === 'rendered') {
    self.pyodide.runPythonAsync(`
    from panel.io.state import state
    from panel.io.pyodide import _link_docs_worker

    _link_docs_worker(state.curdoc, sendPatch, setter='js')
    `)
  } else if (msg.type === 'patch') {
    self.pyodide.globals.set('patch', msg.patch)
    self.pyodide.runPythonAsync(`
    state.curdoc.apply_json_patch(patch.to_py(), setter='js')
    `)
    self.postMessage({type: 'idle'})
  } else if (msg.type === 'location') {
    self.pyodide.globals.set('location', msg.location)
    self.pyodide.runPythonAsync(`
    import json
    from panel.io.state import state
    from panel.util import edit_readonly
    if state.location:
        loc_data = json.loads(location)
        with edit_readonly(state.location):
            state.location.param.update({
                k: v for k, v in loc_data.items() if k in state.location.param
            })
    `)
  }
}

startApplication()