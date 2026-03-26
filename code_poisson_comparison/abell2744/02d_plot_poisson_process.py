import marimo

__generated_with = "0.20.4"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Figures: comparing the likelihood of the Poisson point process for the GC (sub-)samples and the maps
    """)
    return


@app.cell
def _(os):
    # create the output path
    out_path = os.path.join(".", "imgs", "poisson_process")
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    # loop over each GC sample - it defines the number of data points to spawn

    labels = [
        "Bergamini23",
        "Price25",
        "Cha24_WL",
        "Cha24_SL_WL",
        "Original",
        "BCGless",
        "X-ray",
        "uniform",
         "Models",
    ]
    return labels, out_path


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Main program
    """)
    return


@app.cell
def _(read_tables_gcs):
    data_points = read_tables_gcs("Bright GCs")
    data_points.keys()
    return


@app.cell
def _(labels, read_tables_gcs):
    _data_points = read_tables_gcs("Bright GCs")

    for key in labels:
        print(key)  # _data_points[key])
    return


@app.cell
def _(
    labels,
    mo,
    mpl,
    numpy,
    os,
    out_path,
    plt,
    read_tables_gcs,
    read_tables_models,
):
    def figure_lnP(out_path=out_path, labels=labels):
        fig, axs = plt.subplots(2, 2, figsize=(16, 12), sharey=True)
        axs = axs.ravel()

        for j, gcs_name, gcs_label in zip(
            [0, 1, 2, 3],
            ["Bright GCs", "Bright Blue GCs", "Bright Red GCs", "High-quality GCs"],
            [
                "F150W$ < 29.5$",
                "F150W<29.5\n(F115W-F200W)$_0 < 0$",
                "F150W<29.5\n(F115W-F200W)$_0 > 0$",
                "F150W<29.5\nZone 1 and 2",
            ],
        ):
            axs[j].annotate(
                gcs_label, xy=(0.98, 0.02), xycoords="axes fraction", ha="right"
            )
            print(f"\nREADING - {gcs_name}")

            # read the data
            data_points = read_tables_gcs(gcs_name)
            data_maps = read_tables_models(gcs_name)

            for i, key in enumerate(labels):
                # try:
                axs[j].scatter(
                    data_points[key], i, marker="*", color="C{:d}".format(i), s=200
                )
                # except: print(f"WARNING: Observational - {key} not ready yet")

                try:
                    add_violin_plot(
                        axs[j],
                        i,
                        data_maps[f"{key}-{key}"],
                        mean=0,
                        sigma=1,
                        color=f"C{i}",
                    )

                except:
                    print(f"WARNING: Self-maps - {key} not ready yet")

        lgd_elements = [
            mpl.lines.Line2D(
                [0], [0], marker="*", ls="", c="k", markerfacecolor="k", markersize=10
            ),
            mpl.lines.Line2D(
                [0],
                [0],
                marker="*",
                ls="",
                c="None",
                markeredgecolor="k",
                markersize=10,
            ),
        ]

        for i, ax in enumerate(axs):
            ax.set_yticks(numpy.arange(len(labels)))
            if i == 0:
                ax.set_yticklabels(labels)
            ax.invert_yaxis()
            ax.axhline(3.5, ls=":", c="k", lw=0.5)
            ax.axhline(5.5, ls=":", c="k", lw=0.5)
            ax.axhline(6.5, ls=":", c="k", lw=0.5)
            ax.set_xlabel(r"$\ln \mathcal{P}$")
        left = 0.1
        right = 0.98
        top = 0.9
        bottom = 0.1
        hspace = 0.0
        wspace = 0.05
        fig.subplots_adjust(
            left=left, top=top, bottom=bottom, right=right, hspace=hspace, wspace=wspace
        )

        fname = os.path.join(out_path, "poisson_process_lnP.png")
        fig.savefig(fname, dpi=300, bbox_inches="tight")
        return fig

    _fig = figure_lnP(out_path=out_path, labels=labels)
    mo.md(f"""Here's the plot!{mo.as_html(_fig)}""")
    return


@app.cell
def _(
    labels,
    mo,
    mpl,
    numpy,
    os,
    out_path,
    plt,
    read_tables_gcs,
    read_tables_models,
):
    def figure_lnP_per_model(out_path=out_path, labels=labels):
        fig, axs = plt.subplots(3, 3, figsize=(16, 10), sharey=True)
        axs = axs.ravel()

        gc_samples = [
            "Bright GCs",
            "Bright Blue GCs",
            "Bright Red GCs",
            "High-quality GCs",
        ]
        gc_colors = ["k", "C0", "C3", "C2"]

        for j, key in enumerate(labels):
            axs[j].annotate(key, xy=(0.98, 0.02), xycoords="axes fraction", ha="right")

            for i, gcs_name, color in zip(
                range(len(gc_samples)), gc_samples, gc_colors
            ):
                data_points = read_tables_gcs(gcs_name)
                data_maps = read_tables_models(gcs_name)

                try:
                    axs[j].scatter(data_points[key], i, marker="*", color=color, s=200)
                except:
                    print(f"WARNING: Observational - {key} not ready yet")

                try:
                    add_violin_plot(
                        axs[j],
                        i,
                        data_maps[f"{key}-{key}"],
                        mean=0,
                        sigma=1,
                        color=color,
                    )
                except:
                    print(f"WARNING: Self-maps - {key} not ready yet")

        lgd_elements = [
            mpl.lines.Line2D(
                [0], [0], marker="*", ls="", c="k", markerfacecolor="k", markersize=10
            ),
            mpl.lines.Line2D(
                [0],
                [0],
                marker="*",
                ls="",
                c="None",
                markeredgecolor="k",
                markersize=10,
            ),
        ]

        for i, ax in enumerate(axs):
            ax.set_yticks(numpy.arange(len(gc_samples)))
            if i == 0:
                ax.set_yticklabels(gc_samples)
            ax.set_xlabel(r"$\ln \mathcal{P}$")  # /(\ln\mathcal{P})_{\rm max}
        left = 0.1
        right = 0.98
        top = 0.93
        bottom = 0.1
        hspace = 0.25
        wspace = 0.05
        fig.subplots_adjust(
            left=left, top=top, bottom=bottom, right=right, hspace=hspace, wspace=wspace
        )

        fname = os.path.join(out_path, "poisson_process_lnP_model.png")
        fig.savefig(fname, dpi=300, bbox_inches="tight")
        return fig

    _fig = figure_lnP_per_model(out_path=out_path)
    mo.md(f"""Here's the plot!{mo.as_html(_fig)}""")
    return


@app.cell
def _(mo):
    mo.md("""
    mo.md(r)
    """)
    return


@app.cell
def _(
    AutoMinorLocator,
    labels,
    mo,
    mpl,
    numpy,
    os,
    out_path,
    plt,
    read_tables_gcs,
    read_tables_models,
):
    def fig_Zscore_per_gc_sample(out_path=out_path, labels=labels):
        fig, axs = plt.subplots(1, figsize=(7, 5.5), sharex=True, sharey=True)
        axs = numpy.atleast_1d(axs)
        axs = axs.ravel()

        # gc_samples = ["Bright GCs"]
        gc_samples = ["Bright GCs"]

        for j, gcs_name in enumerate(gc_samples):
            axs[j].annotate(
                gcs_name,
                xy=(0.98, 0.95),
                xycoords="axes fraction",
                ha="right",
                va="top",
            )

            # read the input tables
            data_points = read_tables_gcs(gcs_name)
            data_maps = read_tables_models(gcs_name)

            print(f"\nREADING {gcs_name}")

            for i, key in enumerate(labels):
                try:
                    mean = numpy.mean(data_maps[f"{key}-{key}"])
                    sigma = numpy.std(data_maps[f"{key}-{key}"])
                    add_violin_plot(
                        axs[j], i, data_maps[f"{key}-{key}"], mean, sigma, f"C{i}"
                    )
                except:
                    print(f"WARNING: Self-map - {key} is not ready yet")
                    continue

                # show the observational cases
                try:
                    axs[j].scatter(
                        (data_points[key] - mean) / sigma,
                        i - 0.2,
                        marker="*",
                        color="C{:d}".format(i),
                        s=200,
                    )
                except:
                    print(f"WARNING: Observational - {key} not ready yet")

        lgd_elements = [
            mpl.lines.Line2D(
                [0], [0], marker="*", ls="", c="k", markerfacecolor="k", markersize=10
            ),
            mpl.lines.Line2D(
                [0],
                [0],
                marker="*",
                ls="",
                c="None",
                markeredgecolor="k",
                markersize=10,
            ),
        ]

        for i, ax in enumerate(axs):
            ax.set_yticks(numpy.arange(len(labels)))
            ax.set_ylim(-0.8, len(labels) - 0.8)
            ax.invert_yaxis()
            ax.tick_params(
                direction="in",
                which="both",
                bottom=True,
                top=True,
                left=True,
                right=True,
            )
            # set minor ticks in xaxis
            ax.xaxis.set_minor_locator(AutoMinorLocator())
            ax.set_xlim(-5, 50)
            ax.axhline(3.2, ls=":", c="k", lw=0.5)
            # ax.axhline(5.5, ls=":", c="k", lw=0.5)
            # ax.axhline(6.5, ls=":", c="k", lw=0.5)
            if i == 0:
                ax.set_yticklabels(labels)
            ax.set_xlabel(
                "$\\mathcal{Z} = (\\ln \\mathcal{P} - E[\\ln \\mathcal{P}])/\\sigma$"
            )
        left = 0.1
        right = 0.98
        top = 0.9
        bottom = 0.1
        hspace = 0.0
        wspace = 0.05
        fig.subplots_adjust(
            left=left, top=top, bottom=bottom, right=right, hspace=hspace, wspace=wspace
        )
        fname = os.path.join(
            out_path, f"poisson_process_Zscore_{gcs_name}.pdf".replace(" ", "_")
        )
        fig.savefig(fname, dpi=300, bbox_inches="tight")
        # fig.show()
        return fig

    _labels = [
        "Bergamini23",
        "Price25",
        "Cha24_WL",
        "Cha24_SL_WL",
        "Original",
        "Models"
    ]
    _fig = fig_Zscore_per_gc_sample(out_path, _labels)
    mo.md(f""" Here's the plot! {mo.as_html(_fig)} """)
    return


@app.cell
def _(
    labels,
    mo,
    mpl,
    numpy,
    os,
    out_path,
    plt,
    read_tables_gcs,
    read_tables_models,
):
    def fig_Zscore_per_model(out_path=out_path, labels=labels):
        fig, axs = plt.subplots(3, 3, figsize=(16, 10), sharex=False, sharey=True)
        axs = axs.ravel()
        gc_samples = [
            "Bright GCs",
            "Bright Blue GCs",
            "Bright Red GCs",
            "High-quality GCs",
        ]
        gc_colors = ["k", "C0", "C3", "C2"]

        for j, key in enumerate(labels):
            axs[j].annotate(key, xy=(0.98, 0.02), xycoords="axes fraction", ha="right")
            print(f"READING - {key}")
            for i, gcs_name, color in zip(
                range(len(gc_samples)), gc_samples, gc_colors
            ):
                data_maps = read_tables_models(gcs_name)
                data_points_obs = read_tables_gcs(gcs_name)

                # show the violin plots of the self map comparison
                try:
                    mean = numpy.mean(data_maps[f"{key}-{key}"])
                    sigma = numpy.std(data_maps[f"{key}-{key}"])
                    add_violin_plot(
                        axs[j],
                        i,
                        data_maps[f"{key}-{key}"],
                        mean,
                        sigma=sigma,
                        color=color,
                    )
                except:
                    print(f"{key}-{key} NOT ready yet")
                    continue

                # show the observational cases
                try:
                    axs[j].scatter(
                        (data_points_obs[key] - mean) / sigma,
                        i,
                        marker="*",
                        color=color,
                        s=200,
                    )
                except:
                    print(f"WARNING: Observational - {key} not ready yet")

        lgd_elements = [
            mpl.lines.Line2D(
                [0], [0], marker="*", ls="", c="k", markerfacecolor="k", markersize=10
            ),
            mpl.lines.Line2D(
                [0],
                [0],
                marker="*",
                ls="",
                c="None",
                markeredgecolor="k",
                markersize=10,
            ),
        ]

        for i, ax in enumerate(axs):
            ax.set_yticks(numpy.arange(len(gc_samples)))
            if i == 0:
                ax.set_yticklabels(gc_samples)
            ax.invert_yaxis()
            if i >= 6:
                ax.set_xlabel(
                    "$\\mathcal{Z} = (\\ln \\mathcal{P} - E[\\ln \\mathcal{P}])/\\sigma$"
                )
        left = 0.1
        right = 0.98
        top = 0.95
        bottom = 0.1
        hspace = 0.2
        wspace = 0.05
        fig.subplots_adjust(
            left=left, top=top, bottom=bottom, right=right, hspace=hspace, wspace=wspace
        )
        fname = os.path.join(out_path, "poisson_process_Zscore_per_model.pdf")
        fig.savefig(fname, dpi=300, bbox_inches="tight")
        # fig.show()
        return fig

    _fig = fig_Zscore_per_model(out_path, labels)
    mo.md(f""" Here's the plot! {mo.as_html(_fig)} """)
    return


@app.cell
def _(
    AutoMinorLocator,
    labels,
    mo,
    numpy,
    os,
    out_path,
    plt,
    read_tables_gcs,
    read_tables_models,
):
    from ctypes.macholib.dyld import framework_find
    def fig_Zscore_per_selected_model(out_path=out_path, labels=labels):
        fig, axs = plt.subplots(1, 3, figsize=(18, 4.5), sharex=False, sharey=True)
        axs = axs.ravel()
        gc_samples = [
            "Bright GCs",
            "High-quality GCs",
            "Bright Blue GCs",
            "Bright Red GCs",
        ]
        gc_colors = ["k", "C0", "C3", "C6"]

        ls_markers = ["s", "X", "p", "d", "o", ">", "<", "^", "v"]
        for j, key in enumerate(labels):
            print(f"READING - {key}")
            for i, gcs_name, color in zip(
                range(len(gc_samples)), gc_samples, gc_colors
            ):
                data_maps = read_tables_models(gcs_name)
                data_points_obs = read_tables_gcs(gcs_name)

                if "Berg" in key or "Price" in key or "Cha" in key:
                    idx = 0
                    label = "Mass tracers"
                    c = f"C{j}"
                elif "Original" in key or "BCG" in key or "Models" in key:
                    idx = 1
                    label = "Stellar light tracers"
                    c = f"C{j}"
                elif "X-ray" in key:
                    idx = 2
                    label = "X-ray tracer"
                    c = f"C{j}"

                # show the violin plots of the self map comparison
                # try:
                mean = numpy.mean(data_maps[f"{key}-{key}"])
                sigma = numpy.std(data_maps[f"{key}-{key}"])
                add_violin_plot(
                    axs[idx],
                    i,
                    numpy.abs((data_maps[f"{key}-{key}"] - mean) / sigma),
                    mean=0,
                    # sigma=sigma,
                    color=c,
                )
                if i == 0:
                    ll = key
                else:
                    ll = ""
                axs[idx].vlines(numpy.abs((data_points_obs[key] - mean) / sigma), i, i-0.6, ls="-", colors=c, lw=5, label = ll, zorder=100)
                print(key, gcs_name, numpy.abs((data_points_obs[key].value - mean) / sigma))

        for i, ax in enumerate(axs):
            ax.set_yticks(numpy.arange(len(gc_samples)))
            ax.set_xlim(0, 50)
            ax.set_ylim(-0.8, len(gc_samples) - 0.8)
            ax.tick_params(
                direction="in",
                which="both",
                bottom=True,
                top=True,
                left=True,
                right=True,
            )
            # set minor ticks in xaxis
            ax.xaxis.set_minor_locator(AutoMinorLocator())
            if i == 0:
                labels = gc_samples.copy()
                labels[1] = "Complete GCs"
                ax.set_yticklabels(labels)
            ax.invert_yaxis()
            ax.set_xlabel(
                "$|\\mathcal{Z}| = |(\\ln \\mathcal{P} - E[\\ln \\mathcal{P}])/\\sigma|$"
            )
            ax.legend(loc="lower center", ncols=2, bbox_to_anchor=(0.5, 1.00), frameon = False)

        axs[0].annotate(
            "Mass tracers",
            xy=(0.98, 0.95),
            xycoords="axes fraction",
            ha="right",
            va="top",
        )
        axs[1].annotate(
            "Stellar light",
            xy=(0.98, 0.95),
            xycoords="axes fraction",
            ha="right",
            va="top",
        )
        axs[2].annotate(
            "X-ray", xy=(0.98, 0.95), xycoords="axes fraction", ha="right", va="top"
        )

        for ax in axs:
            ax.axhline(0.2, ls=":", c="k", lw=0.5)
            ax.axhline(1.2, ls=":", c="k", lw=0.5)

        left = 0.1
        right = 0.98
        top = 0.95
        bottom = 0.1
        hspace = 0.2
        wspace = 0.08
        fig.subplots_adjust(
            left=left, top=top, bottom=bottom, right=right, hspace=hspace, wspace=wspace
        )
        fname = os.path.join(out_path, "poisson_process_Zscore_per_selected_model.pdf")
        fig.savefig(fname, dpi=300, bbox_inches="tight")
        # fig.show()
        return fig

    _labels = [
        "Bergamini23",
        "Price25",
        "Cha24_WL",
        "Cha24_SL_WL",
        "Original",
        "Models",
        "X-ray",
    ]
    _fig = fig_Zscore_per_selected_model(out_path, _labels)
    mo.md(f""" Here's the plot! {mo.as_html(_fig)} """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Self & cross-map comparison
    """)
    return


@app.cell
def _(
    AutoMinorLocator,
    mo,
    numpy,
    os,
    out_path,
    plt,
    read_tables_gcs,
    read_tables_models,
):
    def fig_Zscore_model_comparison_self_cross(out_path, models):
        #fig, axs = plt.subplots(4, 1, figsize=(7, 18), sharex=False, sharey=False)
        fig, axs = plt.subplots(2, 2, figsize=(12, 10), sharex=False, sharey=False)

        axs = axs.ravel()

        # read the input tables
        data_maps = read_tables_models("Bright GCs")
        data_points_obs = read_tables_gcs("Bright GCs")

        gc_samples = ["Bright GCs"]
        gc_colors = ["k"]

        # set a model for lambda2
        for j, key in enumerate(models):
            labels = []
            idx = 0
            axs[j].annotate(
                f"$\\lambda_2$ = {key}",
                xy=(0.95, 0.95),
                xycoords="axes fraction",
                ha="right",
                va="top",
                fontsize=20,
            )

            # Self-map comparison
            for _k in range(4):
                try:
                    mean = numpy.mean(data_maps[f"{key}-{key}"])
                    sigma = numpy.std(data_maps[f"{key}-{key}"])
                    add_violin_plot(
                        axs[j],
                        _k,
                        numpy.abs((data_maps[f"{key}-{key}"] - mean) / sigma),
                        mean=0,
                        color="k",
                    )
                except:
                    print(f"{key}-{key} NOT ready yet")
                    # mean = 0; sigma = 1
                    # continue

            # idx += 1
            # labels.append("Itself")

            try:
                axs[j].scatter(
                    numpy.abs((data_points_obs[key] - mean) / sigma),
                    idx - 0.3,
                    marker="*",
                    color="black",
                    s=200,
                    lw=2,
                )
            except:
                print(f"{key} - OBS NOT ready yet")
            labels.append("Bright GCs")
            idx += 1

            # Cross-map comparison
            for i, key2 in enumerate(models):
                if key2 == key:
                    continue
                labels.append(key2)
                try:
                    add_violin_plot(
                        axs[j],
                        idx,
                        numpy.abs((data_maps[f"{key2}-{key}"] - mean) / sigma),
                        mean=0,
                        color=f"C{i}",
                    )
                except:
                    print(f"{key2}-{key} NOT ready yet")
                idx += 1

            # format the axes
            axs[j].set_yticks(numpy.arange(len(labels)))
            axs[j].set_yticklabels(labels)
            axs[j].invert_yaxis()
            axs[j].tick_params(axis='y', rotation=55)
            axs[j].axhline(0.2, c="k", lw=0.5, ls="--")
            # axs[j].axvline(0, c="k", lw=0.5, ls=":")
            axs[j].tick_params(
                direction="in",
                which="both",
                bottom=True,
                top=True,
                left=True,
                right=True,
            )
            # set minor ticks in xaxis
            axs[j].xaxis.set_minor_locator(AutoMinorLocator())
            axs[j].set_xlim(0, 70)
        axs[-2].set_xlabel(
            "$|\\mathcal{Z}| = |(\\ln \\mathcal{P} - E[\\ln \\mathcal{P}])/\\sigma|$"
        )
        axs[-1].set_xlabel(
            "$|\\mathcal{Z}| = |(\\ln \\mathcal{P} - E[\\ln \\mathcal{P}])/\\sigma|$"
        )

        left = 0.1
        right = 0.98
        top = 0.95
        bottom = 0.1
        hspace = 0.1
        wspace = 0.25
        fig.subplots_adjust(
            left=left, top=top, bottom=bottom, right=right, hspace=hspace, wspace=wspace
        )
        fname = os.path.join(out_path, "poisson_process_Zscore_self_cross_models.pdf")
        fig.savefig(fname, dpi=300, bbox_inches="tight")
        return fig

    _labels = [
        "Price25",
        "Cha24_WL",
        "Original",
        "X-ray",
        # "uniform",
    ]

    _fig = fig_Zscore_model_comparison_self_cross(out_path, _labels)
    mo.md(f""" Here's the plot! {mo.as_html(_fig)} """)
    return


@app.cell
def _(
    AutoMinorLocator,
    mo,
    numpy,
    os,
    out_path,
    plt,
    read_tables_gcs,
    read_tables_models,
):
    def fig_Zscore_model_comparison_self_cross_collapsed(out_path, models):
        fig, axs = plt.subplots(4, 1, figsize=(7, 9), sharex=False, sharey=False)

        axs = axs.ravel()

        # read the input tables
        data_maps = read_tables_models("Bright Blue GCs")
        data_points_obs = read_tables_gcs("Bright Blue GCs")

        gc_samples = ["Bright Blue GCs"]
        gc_colors = ["C0"]

        # set a model for lambda2
        for j, key in enumerate(models):
            # Self-map comparison
            mean = numpy.mean(data_maps[f"{key}-{key}"])
            sigma = numpy.std(data_maps[f"{key}-{key}"])
            add_violin_plot(
                axs[j],
                0,
                numpy.abs((data_maps[f"{key}-{key}"] - mean) / sigma),
                mean=0,
                color="black",
            )

            #axs[j].scatter(
            #    numpy.abs((data_points_obs[key] - mean) / sigma),
            #    -0.3,
            #    marker="*",
            #    color="C0", edgecolor = "k",
            #    s=300,
            #    lw=0.3, zorder =100
            #)
            axs[j].vlines(numpy.abs((data_points_obs[key] - mean) / sigma), 0, -0.6, ls="-", colors="C0", lw=5, zorder=100)

            axs[j].annotate("Bright Blue GCs", xy=(0.02, 0.95), xycoords="axes fraction", ha="left", va="top", color = "C0")
            # Cross-map comparison
            k = 0
            for i, key2 in enumerate(models):
                if key2 == key:
                    continue
                xxmean = numpy.mean(numpy.abs((data_maps[f"{key2}-{key}"] - mean) / sigma))
                add_violin_plot(
                    axs[j],
                    0,
                    numpy.abs((data_maps[f"{key2}-{key}"] - mean) / sigma),
                    mean=0,
                    color=f"C{i+1}",
                )
                axs[j].annotate(key2, xy=(0.45+0.2*k, 0.95), xycoords="axes fraction", ha="center", va="top", color = f"C{i+1}")
                k += 1

            # format the axes
            axs[j].set_yticks([-0.8])
            axs[j].set_yticklabels([""])
            axs[j].set_ylabel(f"$\\lambda_2$ = {key}")
            axs[j].invert_yaxis()
            axs[j].tick_params(
                direction="in",
                which="both",
                bottom=True,
                top=True,
                left=False,
                right=False,
            )
            # set minor ticks in xaxis
            axs[j].xaxis.set_minor_locator(AutoMinorLocator())
            axs[j].set_xlim(0, 50)
        axs[-1].set_xlabel(
            "$|\\mathcal{Z}| = |(\\ln \\mathcal{P} - E[\\ln \\mathcal{P}])/\\sigma|$"
        )

        left = 0.1
        right = 0.98
        top = 0.95
        bottom = 0.1
        hspace = 0.25
        wspace = 0.05
        fig.subplots_adjust(
            left=left, top=top, bottom=bottom, right=right, hspace=hspace, wspace=wspace
        )
        fname = os.path.join(out_path, "poisson_process_Zscore_self_cross_models_collapsed_blue.pdf")
        fig.savefig(fname, dpi=300, bbox_inches="tight")
        return fig

    _labels = [
        "Price25",
        "Cha24_WL",
        "Models",
        "X-ray"]

    _fig = fig_Zscore_model_comparison_self_cross_collapsed(out_path, _labels)
    mo.md(f""" Here's the plot! {mo.as_html(_fig)} """)
    return


@app.cell
def _(
    mo,
    mpl,
    numpy,
    os,
    out_path,
    pandas,
    plt,
    read_tables_gcs,
    read_tables_models,
    sns,
):
    def fig_Zscore_model_comparison_self_cross_all(out_path, models, fname):
        dict_results = {}  # "Models" : []}# = numpy.zeros((len(models), len(models)))
        # lambda 2
        for j, key in enumerate(models):
            data_maps = read_tables_models("Bright Blue GCs")

            # define the baseline for the comparison
            mean2 = numpy.mean(data_maps[f"{key}-{key}"])
            sigma2 = numpy.std(data_maps[f"{key}-{key}"])

            dict_results[key] = []
            # dict_results["Models"].append(key) # save the model name in the first column

            # first: add the GCs
            data_points_obs = read_tables_gcs("Bright Blue GCs")
            dict_results[key].append(
                numpy.abs((data_points_obs[key][0] - mean2) / sigma2)
            )  # save how many sigmas away the mean of the cross-map is from the GCs

            for i, key1 in enumerate(models):
                mean1 = numpy.mean(data_maps[f"{key1}-{key}"])
                dict_results[key].append(
                    numpy.abs((mean1 - mean2) / sigma2)
                )  # save how many sigmas away the mean of the cross-map is from the self-map

        results = pandas.DataFrame(dict_results, index=["Bright Blue GCs"] + models)
        print(results)
        fig = plt.figure(figsize=(22, 16))
        ax = sns.heatmap(
            results,
            annot=True,
            square=True,
            vmin=0,
            vmax=20,
            cmap="Greys_r",
            linewidths=0.5,
            xticklabels = True,
            yticklabels = True,
            cbar_kws={
                "label": r"$|\mathcal{Z}|= |(E[\ln \mathcal{P}\{\lambda_1\}] - E[\ln \mathcal{P}\{\lambda_2\}])/\sigma_{{P}\{\lambda_2\}}|$",
                "fraction" : 0.05, "pad": 0.01, "aspect" : 40, 

            },
        )
        # add xticks at the top and bottom
        ax.tick_params(axis="x", bottom=True, top=True, labelbottom=True, labeltop=True)

        ax.add_patch(
            mpl.patches.Rectangle(
                (1, 1), 1, len(models),
                fill=False,
                edgecolor="C1",
                lw=10,
            ))
        ax.add_patch(
            mpl.patches.Rectangle(
                (6, 1), 1, len(models),
                fill=False,
                edgecolor="C2",
                lw=10,
            )
        )
        ax.add_patch(
            mpl.patches.Rectangle(
                (0, 0), len(models), 1,
                fill=False,
                edgecolor="C0",
                lw=10,
            )
        )
        ax.add_patch(
            mpl.patches.Rectangle(
                (0, 6), len(models), 1,
                fill=False,
                edgecolor="C4",
                lw=10,
            )
        )
        ax.set(xlabel="Map against we compare: $\\lambda_2$", ylabel="Map from which we spawn: $\\lambda_1$")
        #ax.xaxis.tick_top()

        fig.savefig(fname, dpi=300, bbox_inches="tight")
        return fig

    _labels = [
        "Bergamini23",
        "Price25",
        "Cha24_WL",
        "Cha24_SL_WL",
        "Original",
        "Models",
        "X-ray",
    ]

    fname = os.path.join(out_path, "heatmap_Zscore_all_models_blue.pdf")
    _fig = fig_Zscore_model_comparison_self_cross_all(out_path, _labels, fname)
    mo.md(f""" Here's the plot! {mo.as_html(_fig)} """)
    return (fig_Zscore_model_comparison_self_cross_all,)


@app.cell
def _(
    mo,
    mpl,
    numpy,
    os,
    out_path,
    pandas,
    plt,
    read_tables_gcs,
    read_tables_models,
    sns,
):
    def fig_Zscore_model_comparison_self_cross_selected(out_path, models, fname):
        dict_results = {}  # "Models" : []}# = numpy.zeros((len(models), len(models)))
        # lambda 2
        for j, key in enumerate(models):
            data_maps = read_tables_models("Bright Blue GCs")

            # define the baseline for the comparison
            mean2 = numpy.mean(data_maps[f"{key}-{key}"])
            sigma2 = numpy.std(data_maps[f"{key}-{key}"])

            dict_results[key] = []
            # dict_results["Models"].append(key) # save the model name in the first column

            # first: add the GCs
            data_points_obs = read_tables_gcs("Bright Blue GCs")
            dict_results[key].append(
                numpy.abs((data_points_obs[key][0] - mean2) / sigma2)
            )  # save how many sigmas away the mean of the cross-map is from the GCs

            ls_selected = []
            for i, key1 in enumerate(models):
                if "Berg" in key1 or "Price" in key1 or "Cha" in key1: # no need to do the mass maps as tracers
                    continue
                ls_selected.append(key1)
                mean1 = numpy.mean(data_maps[f"{key1}-{key}"])
                dict_results[key].append(
                    numpy.abs((mean1 - mean2) / sigma2)
                )  # save how many sigmas away the mean of the cross-map is from the self-map

        print(ls_selected)

        results = pandas.DataFrame(dict_results, index=["Bright Blue GCs"] + ls_selected)
        print(results)
        fig = plt.figure(figsize=(22, 16))
        ax = sns.heatmap(
            results,
            annot=True,
            square=True,
            vmin=0,
            vmax=20,
            cmap="Greys_r",
            linewidths=0.5,
            xticklabels = True,
            yticklabels = True,
            cbar_kws={
                "label": r"$|\mathcal{Z}|= |(E[\ln \mathcal{P}\{\lambda_1\}] - E[\ln \mathcal{P}\{\lambda_2\}])/\sigma_{{P}\{\lambda_2\}}|$",
                "fraction" : 0.02, "pad": 0.01, "aspect" : 40, "shrink" : 0.75

            },
        )
        # add xticks at the top and bottom
        ax.tick_params(axis="x", bottom=True, top=True, labelbottom=True, labeltop=True)

        ax.add_patch(
            mpl.patches.Rectangle(
                (1, 1), 1, len(ls_selected),
                fill=False,
                edgecolor="C1",
                lw=10,
            ))
        ax.add_patch(
            mpl.patches.Rectangle(
                (6, 1), 1, len(ls_selected),
                fill=False,
                edgecolor="C2",
                lw=10,
            )
        )
        ax.add_patch(
            mpl.patches.Rectangle(
                (0, 0), len(models), 1,
                fill=False,
                edgecolor="C0",
                lw=10,
            )
        )
        ax.add_patch(
            mpl.patches.Rectangle(
                (0, 2), len(models), 1,
                fill=False,
                edgecolor="C4",
                lw=10,
            )
        )
        ax.set(xlabel="Map against we compare ($\\lambda_2$)", ylabel="Tracer: map from which we spawn ($\\lambda_1$)")
        #ax.xaxis.tick_top()

        fig.savefig(fname, dpi=300, bbox_inches="tight")
        return fig

    _labels = [
        "Bergamini23",
        "Price25",
        "Cha24_WL",
        "Cha24_SL_WL",
        "Original",
        "Models",
        "X-ray",
    ]

    _fname = os.path.join(out_path, "heatmap_Zscore_all_models_blue_selected.pdf")
    _fig = fig_Zscore_model_comparison_self_cross_selected(out_path, _labels, _fname)
    mo.md(f""" Here's the plot! {mo.as_html(_fig)} """)
    return


@app.cell
def _():
    return


@app.cell
def _(fig_Zscore_model_comparison_self_cross_all, mo, os, out_path):

    _labels = [
        "Price25",
        "Cha24_WL",
        "Original",
        "X-ray",
        "uniform",
        "Models"
    ]
    _fname = os.path.join(out_path, "heatmap_Zscore_selected_models.pdf")
    _fig = fig_Zscore_model_comparison_self_cross_all(out_path, _labels, _fname)
    mo.md(f""" Here's the plot! {mo.as_html(_fig)} """)
    return


@app.cell
def _(fig_Zscore_model_comparison_self_cross_all, mo, os, out_path):
    _labels = [
        "compact",
        "extended",
    ]
    _fname = os.path.join(out_path, "heatmap_Zscore_compact_extended.pdf")
    _fig = fig_Zscore_model_comparison_self_cross_all(out_path, _labels, _fname)

    mo.md(f""" Here's the plot! {mo.as_html(_fig)} """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Functions
    """)
    return


@app.cell
def _(ascii, os):
    def read_tables_gcs(gcs_name):
        # read the output table of the point - map comparison
        try:
            table = os.path.join(
                ".",
                "tables",
                "points_to_maps",
                f"table_{gcs_name}.ecsv".replace(" ", "_"),
            )
            return_data = ascii.read(table, format="ecsv")
        except:
            print(f"WARNING - table {table} not ready yet")
            return_data = None

        return return_data

    return (read_tables_gcs,)


@app.cell
def _(ascii, glob, hstack, os):
    def read_tables_models(gcs_name):
        try:
            # read the output table of the map - map comparison
            tnames = glob.glob(
                os.path.join(
                    ".",
                    "tables",
                    "maps_to_maps",
                    f"table_{gcs_name}_testing_*.ecsv".replace(" ", "_"),
                )
            )
            dummy_data = []
            for name in tnames:
                dummy_data.append(ascii.read(name, format="ecsv"))
            return_data = hstack(tables=dummy_data)
        except:
            print(f"WARNING - tables are not ready yet")
            return_data = None

        return return_data

    return (read_tables_models,)


@app.function
def add_violin_plot(ax, idx, data, mean, sigma=1, color="k"):
    parts = ax.violinplot(
        (data - mean) / sigma,
        [idx],
        points=200,
        orientation="horizontal",
        widths=1.2,
        showmedians=False,
        showextrema=False,
        bw_method=0.5,
        side="low",
    )
    # make the violin bodies the same colour
    for pc in parts["bodies"]:
        pc.set_facecolor(color)
        pc.set_edgecolor("black")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Modules
    """)
    return


@app.cell
def _():
    import marimo as mo

    # Import modules
    import numpy, os, glob, pandas
    import matplotlib.pyplot as plt
    plt.style.use('petroff10')
    import matplotlib as mpl
    from astropy.table import Table, hstack
    from astropy.io import ascii
    import seaborn as sns
    from matplotlib.ticker import AutoMinorLocator

    mpl.rcParams["text.usetex"] = False
    mpl.rcParams["font.size"] = 18.0
    mpl.rcParams["legend.fontsize"] = 16.0
    return (
        AutoMinorLocator,
        ascii,
        glob,
        hstack,
        mo,
        mpl,
        numpy,
        os,
        pandas,
        plt,
        sns,
    )


if __name__ == "__main__":
    app.run()
