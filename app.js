/**
 * Interactive TPU Hardware & Grouped Matmul (gmm / ragged_dot) Simulator
 * Uses strictly safe DOM creation APIs (createElementNS / textContent / replaceChildren)
 * with zero innerHTML and zero inline style attributes for strict CSP compliance.
 */

(function () {
  'use strict';

  const SVG_NS = 'http://www.w3.org/2000/svg';
  const TILE_SIZE = 128;

  const COLORS = {
    ink: '#181c25',
    muted: '#5e6573',
    panelBg: '#ffffff',
    panelStroke: '#dcd8ce',
    subBg: '#f6f3ec',
    dense: '#216e9c',
    e0: '#1c69ce',
    e1: '#d67a10',
    e2: '#1b917c',
    good: '#198558',
    stallFill: '#f8d7d4',
    stallStroke: '#c8372d',
    idleCell: '#e3e1da',
    activeHighlight: '#181c25'
  };

  const state = {
    mode: 'gmm',
    e0: 510,
    e1: 35,
    e2: 290,
    progress: 1,
    rafId: null
  };

  function clampInt(raw, min, max, fallback) {
    const parsed = Number.parseInt(String(raw), 10);
    if (Number.isNaN(parsed)) {
      return fallback;
    }
    return Math.min(max, Math.max(min, parsed));
  }

  function svgEl(tag, attrs) {
    const node = document.createElementNS(SVG_NS, tag);
    if (attrs) {
      const keys = Object.keys(attrs);
      for (let i = 0; i < keys.length; i += 1) {
        const k = keys[i];
        node.setAttribute(k, String(attrs[k]));
      }
    }
    return node;
  }

  function svgText(content, attrs) {
    const node = svgEl('text', attrs);
    node.textContent = String(content);
    return node;
  }

  function computeMetrics(e0, e1, e2) {
    const total = e0 + e1 + e2;
    const maxE = Math.max(e0, e1, e2);

    // 1. Dense FFN baseline (single shared weight matrix DMA + minimal tail)
    const denseCost = total + 42;
    const denseUtil = Math.round((total / denseCost) * 100);

    // 2. MoE Naive #1: Sequential For-Loop (3 kernel launches + tile underfill + dispatch stalls)
    const loopTiledRows =
      Math.ceil(e0 / TILE_SIZE) * TILE_SIZE +
      Math.ceil(e1 / TILE_SIZE) * TILE_SIZE +
      Math.ceil(e2 / TILE_SIZE) * TILE_SIZE;
    const loopStallCost = 3 * 115;
    const loopCost = loopTiledRows + loopStallCost;
    const loopUtil = Math.round((total / loopCost) * 100);
    const minExpert = Math.min(e0, e1, e2);
    const minTileIdlePct = Math.max(
      0,
      Math.round(((TILE_SIZE - Math.min(TILE_SIZE, minExpert)) / TILE_SIZE) * 100)
    );

    // 3. MoE Naive #2: Zero-Padded Batched Matmul ([3, maxE, K])
    const paddedRows = 3 * maxE;
    const zeroRows = paddedRows - total;
    const paddedTiledRows = Math.ceil(paddedRows / TILE_SIZE) * TILE_SIZE;
    const paddedCost = paddedTiledRows + 40;
    const paddedWastePct = paddedRows > 0 ? (zeroRows / paddedRows) * 100 : 0;
    const paddedUtil = Math.round((total / paddedCost) * 100);

    // 4. MoE Grouped Matmul (gmm / ragged_dot Pallas kernel)
    // Sublane VMEM boundary alignment only; zero HBM zero-padding
    const gmmBoundaryOverhead = 48;
    const gmmLaunchOverhead = 42;
    const gmmCost = total + gmmBoundaryOverhead + gmmLaunchOverhead;
    const gmmUtil = Math.round((total / gmmCost) * 100);

    const maxCost = Math.max(denseCost, loopCost, paddedCost, gmmCost);

    return {
      total,
      maxE,
      minExpert,
      minTileIdlePct,
      paddedRows,
      zeroRows,
      paddedWastePct,
      maxCost,
      dense: {
        cost: denseCost,
        util: denseUtil,
        wastePct: 0,
        idlePct: Math.max(0, 100 - denseUtil)
      },
      loop: {
        cost: loopCost,
        util: loopUtil,
        wastePct: 0,
        idlePct: Math.max(0, 100 - loopUtil),
        stallCost: loopStallCost,
        tiledRows: loopTiledRows
      },
      padded: {
        cost: paddedCost,
        util: paddedUtil,
        wastePct: Math.round((zeroRows / paddedCost) * 100),
        hbmWastePct: paddedWastePct,
        idlePct: Math.max(0, 100 - paddedUtil - Math.round((zeroRows / paddedCost) * 100))
      },
      gmm: {
        cost: gmmCost,
        util: gmmUtil,
        wastePct: 0,
        idlePct: Math.max(0, 100 - gmmUtil)
      }
    };
  }

  function buildDefs() {
    const defs = svgEl('defs');

    const hatch = svgEl('pattern', {
      id: 'sim-waste-hatch',
      width: 8,
      height: 8,
      patternTransform: 'rotate(45 0 0)',
      patternUnits: 'userSpaceOnUse'
    });
    hatch.appendChild(svgEl('rect', { width: 8, height: 8, fill: '#fae2e0' }));
    hatch.appendChild(
      svgEl('line', { x1: 0, y1: 0, x2: 0, y2: 8, stroke: '#d14338', 'stroke-width': 2.2 })
    );
    defs.appendChild(hatch);

    const stallHatch = svgEl('pattern', {
      id: 'sim-stall-hatch',
      width: 6,
      height: 6,
      patternTransform: 'rotate(-45 0 0)',
      patternUnits: 'userSpaceOnUse'
    });
    stallHatch.appendChild(svgEl('rect', { width: 6, height: 6, fill: '#ece9e1' }));
    stallHatch.appendChild(
      svgEl('line', { x1: 0, y1: 0, x2: 0, y2: 6, stroke: '#b8b3a6', 'stroke-width': 1.6 })
    );
    defs.appendChild(stallHatch);

    return defs;
  }

  function renderPanelFrame(svg, x, y, w, h, title) {
    svg.appendChild(
      svgEl('rect', {
        x: x,
        y: y,
        width: w,
        height: h,
        rx: 8,
        fill: COLORS.panelBg,
        stroke: COLORS.panelStroke,
        'stroke-width': 1
      })
    );
    svg.appendChild(
      svgText(title, {
        x: x + 16,
        y: y + 26,
        fill: COLORS.muted,
        'font-family': 'JetBrains Mono, monospace',
        'font-size': 10.5,
        'font-weight': 700,
        'letter-spacing': '0.05em'
      })
    );
    svg.appendChild(
      svgEl('line', {
        x1: x + 14,
        y1: y + 36,
        x2: x + w - 14,
        y2: y + 36,
        stroke: '#ebe7de',
        'stroke-width': 1
      })
    );
  }

  function renderMemoryPanel(svg, m) {
    renderPanelFrame(svg, 12, 12, 276, 332, '1. HBM TENSOR LAYOUT');

    const { e0, e1, e2, mode } = state;
    const total = m.total;
    const colX = 28;
    const colW = 114;
    const topY = 62;
    const availH = 210;

    if (mode === 'dense') {
      svg.appendChild(
        svgEl('rect', {
          x: colX,
          y: topY,
          width: colW,
          height: availH,
          rx: 5,
          fill: COLORS.dense
        })
      );
      svg.appendChild(
        svgText('All Tokens', {
          x: colX + colW / 2,
          y: topY + availH / 2 - 6,
          'text-anchor': 'middle',
          fill: '#ffffff',
          'font-family': 'Plus Jakarta Sans, sans-serif',
          'font-size': 12,
          'font-weight': 700
        })
      );
      svg.appendChild(
        svgText('[' + total + ', Hidden]', {
          x: colX + colW / 2,
          y: topY + availH / 2 + 12,
          'text-anchor': 'middle',
          fill: '#dcecf7',
          'font-family': 'JetBrains Mono, monospace',
          'font-size': 10
        })
      );

      svg.appendChild(
        svgText('\u00d7', {
          x: 158,
          y: topY + availH / 2 + 5,
          'text-anchor': 'middle',
          fill: COLORS.ink,
          'font-family': 'JetBrains Mono, monospace',
          'font-size': 16,
          'font-weight': 700
        })
      );

      svg.appendChild(
        svgEl('rect', {
          x: 174,
          y: topY + 34,
          width: 98,
          height: availH - 68,
          rx: 5,
          fill: COLORS.ink
        })
      );
      svg.appendChild(
        svgText('Shared W', {
          x: 223,
          y: topY + availH / 2 - 4,
          'text-anchor': 'middle',
          fill: '#ffffff',
          'font-family': 'Plus Jakarta Sans, sans-serif',
          'font-size': 11.5,
          'font-weight': 700
        })
      );
      svg.appendChild(
        svgText('[H, Interm]', {
          x: 223,
          y: topY + availH / 2 + 13,
          'text-anchor': 'middle',
          fill: '#c8d0dc',
          'font-family': 'JetBrains Mono, monospace',
          'font-size': 9.5
        })
      );

      svg.appendChild(
        svgText('Uniform rectangular matmul (0% pad)', {
          x: 150,
          y: 324,
          'text-anchor': 'middle',
          fill: COLORS.muted,
          'font-family': 'JetBrains Mono, monospace',
          'font-size': 9.8
        })
      );
      return;
    }

    if (mode === 'loop') {
      const gap = 16;
      const usable = availH - gap * 2;
      const h0 = Math.max(18, Math.round((e0 / total) * usable));
      const h1 = Math.max(16, Math.round((e1 / total) * usable));
      const h2 = Math.max(18, usable - h0 - h1);

      const blocks = [
        { label: 'E0 [' + e0 + ', K]', wLabel: 'W_0 [K, N]', h: h0, color: COLORS.e0 },
        { label: 'E1 [' + e1 + ', K]', wLabel: 'W_1 [K, N]', h: h1, color: COLORS.e1 },
        { label: 'E2 [' + e2 + ', K]', wLabel: 'W_2 [K, N]', h: h2, color: COLORS.e2 }
      ];

      let curY = topY;
      for (let i = 0; i < blocks.length; i += 1) {
        const b = blocks[i];
        svg.appendChild(
          svgEl('rect', { x: colX, y: curY, width: colW, height: b.h, rx: 4, fill: b.color })
        );
        svg.appendChild(
          svgText(b.label, {
            x: colX + colW / 2,
            y: curY + b.h / 2 + 3.5,
            'text-anchor': 'middle',
            fill: '#ffffff',
            'font-family': 'JetBrains Mono, monospace',
            'font-size': 9.5,
            'font-weight': 600
          })
        );

        svg.appendChild(
          svgText('\u2192', {
            x: 158,
            y: curY + b.h / 2 + 4,
            'text-anchor': 'middle',
            fill: COLORS.muted,
            'font-family': 'JetBrains Mono, monospace',
            'font-size': 12
          })
        );

        svg.appendChild(
          svgEl('rect', {
            x: 174,
            y: curY,
            width: 98,
            height: b.h,
            rx: 4,
            fill: '#2b3140'
          })
        );
        svg.appendChild(
          svgText(b.wLabel, {
            x: 223,
            y: curY + b.h / 2 + 3.5,
            'text-anchor': 'middle',
            fill: '#f1f5f9',
            'font-family': 'JetBrains Mono, monospace',
            'font-size': 9.5
          })
        );

        curY += b.h + gap;
      }

      svg.appendChild(
        svgText('Disjoint per-expert kernel calls', {
          x: 150,
          y: 324,
          'text-anchor': 'middle',
          fill: COLORS.stallStroke,
          'font-family': 'JetBrains Mono, monospace',
          'font-size': 9.8,
          'font-weight': 600
        })
      );
      return;
    }

    if (mode === 'padded') {
      const slotH = 62;
      const slotGap = 12;
      const experts = [
        { name: 'E0', count: e0, color: COLORS.e0 },
        { name: 'E1', count: e1, color: COLORS.e1 },
        { name: 'E2', count: e2, color: COLORS.e2 }
      ];

      for (let i = 0; i < experts.length; i += 1) {
        const ex = experts[i];
        const sy = topY + i * (slotH + slotGap);
        const realH = Math.max(6, Math.round((ex.count / m.maxE) * slotH));
        const padH = slotH - realH;

        svg.appendChild(
          svgEl('rect', {
            x: colX,
            y: sy,
            width: colW,
            height: realH,
            rx: 3,
            fill: ex.color
          })
        );

        if (padH > 2) {
          svg.appendChild(
            svgEl('rect', {
              x: colX,
              y: sy + realH,
              width: colW,
              height: padH,
              rx: 2,
              fill: 'url(#sim-waste-hatch)',
              stroke: '#d14338',
              'stroke-width': 0.8
            })
          );
          if (padH >= 18) {
            svg.appendChild(
              svgText('+' + (m.maxE - ex.count) + ' zeros', {
                x: colX + colW / 2,
                y: sy + realH + padH / 2 + 3.5,
                'text-anchor': 'middle',
                fill: '#991b1b',
                'font-family': 'JetBrains Mono, monospace',
                'font-size': 9.2,
                'font-weight': 700
              })
            );
          }
        }

        if (realH >= 16) {
          svg.appendChild(
            svgText(ex.name + ': ' + ex.count, {
              x: colX + 8,
              y: sy + Math.min(realH - 4, 14),
              fill: '#ffffff',
              'font-family': 'JetBrains Mono, monospace',
              'font-size': 9.2,
              'font-weight': 600
            })
          );
        }
      }

      svg.appendChild(
        svgText('\u00d7', {
          x: 158,
          y: topY + availH / 2 + 5,
          'text-anchor': 'middle',
          fill: COLORS.ink,
          'font-family': 'JetBrains Mono, monospace',
          'font-size': 16,
          'font-weight': 700
        })
      );

      svg.appendChild(
        svgEl('rect', {
          x: 174,
          y: topY + 26,
          width: 98,
          height: availH - 52,
          rx: 5,
          fill: '#2b3140'
        })
      );
      svg.appendChild(
        svgText('Weights', {
          x: 223,
          y: topY + availH / 2 - 6,
          'text-anchor': 'middle',
          fill: '#ffffff',
          'font-family': 'Plus Jakarta Sans, sans-serif',
          'font-size': 11.5,
          'font-weight': 700
        })
      );
      svg.appendChild(
        svgText('[3, K, N]', {
          x: 223,
          y: topY + availH / 2 + 12,
          'text-anchor': 'middle',
          fill: '#cbd5e1',
          'font-family': 'JetBrains Mono, monospace',
          'font-size': 10
        })
      );

      svg.appendChild(
        svgText('Padded [3, ' + m.maxE + ', K] \u2192 ' + m.zeroRows + ' zero rows!', {
          x: 150,
          y: 324,
          'text-anchor': 'middle',
          fill: '#b42318',
          'font-family': 'JetBrains Mono, monospace',
          'font-size': 9.5,
          'font-weight': 600
        })
      );
      return;
    }

    // Mode: gmm (Grouped Matmul)
    const packedH = 182;
    const h0 = Math.max(16, Math.round((e0 / total) * packedH));
    const h1 = Math.max(14, Math.round((e1 / total) * packedH));
    const h2 = Math.max(16, packedH - h0 - h1);

    const segments = [
      { label: 'E0: ' + e0, h: h0, color: COLORS.e0 },
      { label: 'E1: ' + e1, h: h1, color: COLORS.e1 },
      { label: 'E2: ' + e2, h: h2, color: COLORS.e2 }
    ];

    let py = topY;
    for (let i = 0; i < segments.length; i += 1) {
      const seg = segments[i];
      svg.appendChild(
        svgEl('rect', {
          x: colX,
          y: py,
          width: colW,
          height: seg.h,
          fill: seg.color
        })
      );
      svg.appendChild(
        svgText(seg.label, {
          x: colX + colW / 2,
          y: py + seg.h / 2 + 3.5,
          'text-anchor': 'middle',
          fill: '#ffffff',
          'font-family': 'JetBrains Mono, monospace',
          'font-size': 9.5,
          'font-weight': 600
        })
      );
      py += seg.h;
    }

    // Crisp outer frame around contiguous tensor
    svg.appendChild(
      svgEl('rect', {
        x: colX,
        y: topY,
        width: colW,
        height: packedH,
        rx: 4,
        fill: 'none',
        stroke: COLORS.ink,
        'stroke-width': 1.4
      })
    );

    svg.appendChild(
      svgText('\u00d7', {
        x: 158,
        y: topY + packedH / 2 + 5,
        'text-anchor': 'middle',
        fill: COLORS.ink,
        'font-family': 'JetBrains Mono, monospace',
        'font-size': 16,
        'font-weight': 700
      })
    );

    svg.appendChild(
      svgEl('rect', {
        x: 174,
        y: topY + 24,
        width: 98,
        height: packedH - 48,
        rx: 5,
        fill: COLORS.ink
      })
    );
    svg.appendChild(
      svgText('Stacked W', {
        x: 223,
        y: topY + packedH / 2 - 5,
        'text-anchor': 'middle',
        fill: '#ffffff',
        'font-family': 'Plus Jakarta Sans, sans-serif',
        'font-size': 11.5,
        'font-weight': 700
      })
    );
    svg.appendChild(
      svgText('[3, K, N]', {
        x: 223,
        y: topY + packedH / 2 + 12,
        'text-anchor': 'middle',
        fill: '#cbd5e1',
        'font-family': 'JetBrains Mono, monospace',
        'font-size': 10
      })
    );

    // group_sizes metadata bar
    svg.appendChild(
      svgEl('rect', {
        x: colX,
        y: 258,
        width: 244,
        height: 28,
        rx: 4,
        fill: COLORS.ink
      })
    );
    svg.appendChild(
      svgText('group_sizes = [' + e0 + ', ' + e1 + ', ' + e2 + ']', {
        x: colX + 122,
        y: 276,
        'text-anchor': 'middle',
        fill: '#f8fafc',
        'font-family': 'JetBrains Mono, monospace',
        'font-size': 9.8,
        'font-weight': 600
      })
    );

    svg.appendChild(
      svgText('Packed [' + total + ', K] \u00b7 0 padded rows in HBM', {
        x: 150,
        y: 324,
        'text-anchor': 'middle',
        fill: COLORS.good,
        'font-family': 'JetBrains Mono, monospace',
        'font-size': 9.5,
        'font-weight': 700
      })
    );
  }

  function renderTimelinePanel(svg, m) {
    renderPanelFrame(svg, 300, 12, 326, 332, '2. KERNEL DISPATCH TIMELINE');

    const { e0, e1, e2, mode, progress } = state;
    const trackX = 318;
    const maxTrackW = 290;
    const scale = maxTrackW / m.maxCost;

    const rows = [
      { id: 'dense', label: '1. Dense FFN', y: 66, cost: m.dense.cost },
      { id: 'loop', label: '2. Naive For-Loop', y: 128, cost: m.loop.cost },
      { id: 'padded', label: '3. Zero-Padded Batch', y: 190, cost: m.padded.cost },
      { id: 'gmm', label: '4. Grouped Matmul (gmm)', y: 252, cost: m.gmm.cost }
    ];

    for (let i = 0; i < rows.length; i += 1) {
      const r = rows[i];
      const isSelected = r.id === mode;
      const groupOpacity = isSelected ? 1 : 0.48;
      const g = svgEl('g', { opacity: groupOpacity });

      g.appendChild(
        svgText(r.label, {
          x: trackX,
          y: r.y - 7,
          fill: isSelected ? COLORS.ink : COLORS.muted,
          'font-family': 'Plus Jakarta Sans, sans-serif',
          'font-size': 10.5,
          'font-weight': isSelected ? 700 : 600
        })
      );

      const relSpeed = (r.cost / m.dense.cost).toFixed(2) + '\u00d7 time';
      g.appendChild(
        svgText(relSpeed, {
          x: trackX + maxTrackW,
          y: r.y - 7,
          'text-anchor': 'end',
          fill: isSelected ? COLORS.ink : COLORS.muted,
          'font-family': 'JetBrains Mono, monospace',
          'font-size': 9.5,
          'font-weight': 600
        })
      );

      // Background rail
      g.appendChild(
        svgEl('rect', {
          x: trackX,
          y: r.y,
          width: maxTrackW,
          height: 26,
          rx: 4,
          fill: COLORS.subBg,
          stroke: isSelected ? COLORS.ink : '#e2dfd7',
          'stroke-width': isSelected ? 1.4 : 0.8
        })
      );

      if (r.id === 'dense') {
        const w = Math.max(12, r.cost * scale);
        g.appendChild(
          svgEl('rect', { x: trackX, y: r.y, width: w, height: 26, rx: 4, fill: COLORS.dense })
        );
      } else if (r.id === 'loop') {
        let cx = trackX;
        const stallEach = (m.loop.stallCost / 3) * scale;
        const segs = [
          { count: Math.ceil(e0 / TILE_SIZE) * TILE_SIZE, color: COLORS.e0 },
          { count: Math.ceil(e1 / TILE_SIZE) * TILE_SIZE, color: COLORS.e1 },
          { count: Math.ceil(e2 / TILE_SIZE) * TILE_SIZE, color: COLORS.e2 }
        ];
        for (let s = 0; s < segs.length; s += 1) {
          g.appendChild(
            svgEl('rect', {
              x: cx,
              y: r.y,
              width: Math.max(4, stallEach),
              height: 26,
              fill: COLORS.stallFill
            })
          );
          cx += stallEach;
          const sw = Math.max(6, segs[s].count * scale);
          g.appendChild(
            svgEl('rect', { x: cx, y: r.y, width: sw, height: 26, fill: segs[s].color })
          );
          cx += sw;
        }
      } else if (r.id === 'padded') {
        let cx = trackX + 36 * scale;
        g.appendChild(
          svgEl('rect', { x: trackX, y: r.y, width: 36 * scale, height: 26, fill: '#94a3b8' })
        );
        const exList = [
          { count: e0, color: COLORS.e0 },
          { count: e1, color: COLORS.e1 },
          { count: e2, color: COLORS.e2 }
        ];
        for (let s = 0; s < exList.length; s += 1) {
          const rw = Math.max(3, exList[s].count * scale);
          const pw = Math.max(0, (m.maxE - exList[s].count) * scale);
          g.appendChild(
            svgEl('rect', { x: cx, y: r.y, width: rw, height: 26, fill: exList[s].color })
          );
          cx += rw;
          if (pw > 1) {
            g.appendChild(
              svgEl('rect', {
                x: cx,
                y: r.y,
                width: pw,
                height: 26,
                fill: 'url(#sim-waste-hatch)'
              })
            );
            cx += pw;
          }
        }
      } else if (r.id === 'gmm') {
        const lw = 42 * scale;
        g.appendChild(
          svgEl('rect', { x: trackX, y: r.y, width: lw, height: 26, fill: COLORS.ink })
        );
        let cx = trackX + lw;
        const gSegs = [
          { count: e0 + 16, color: COLORS.e0 },
          { count: e1 + 16, color: COLORS.e1 },
          { count: e2 + 16, color: COLORS.e2 }
        ];
        for (let s = 0; s < gSegs.length; s += 1) {
          const gw = Math.max(5, gSegs[s].count * scale);
          g.appendChild(
            svgEl('rect', { x: cx, y: r.y, width: gw, height: 26, fill: gSegs[s].color })
          );
          cx += gw;
        }
      }

      svg.appendChild(g);
    }

    // Animated Sweep Playhead
    if (progress < 0.999) {
      const playX = trackX + progress * maxTrackW;
      svg.appendChild(
        svgEl('line', {
          x1: playX,
          y1: 54,
          x2: playX,
          y2: 288,
          stroke: COLORS.ink,
          'stroke-width': 2,
          'stroke-dasharray': '3 2'
        })
      );
    }

    svg.appendChild(
      svgText('Wall-clock step time (shorter bar = faster)', {
        x: 463,
        y: 324,
        'text-anchor': 'middle',
        fill: COLORS.muted,
        'font-family': 'JetBrains Mono, monospace',
        'font-size': 9.5
      })
    );
  }

  function renderMxuPanel(svg, m) {
    renderPanelFrame(svg, 638, 12, 270, 332, '3. TPU 128\u00d7128 MXU ARRAY');

    const { e0, e1, e2, mode, progress } = state;
    const total = m.total;
    const totalCells = 64;
    const activeCutoff = Math.round(progress * totalCells);

    const modeData = m[mode];
    const usefulCells = Math.round((modeData.util / 100) * totalCells);
    const wasteCells = Math.round((modeData.wastePct / 100) * totalCells);

    // Split useful cells proportionally across experts in MoE modes
    const c0 = Math.round((e0 / total) * usefulCells);
    const c1 = Math.max(1, Math.round((e1 / total) * usefulCells));

    const gridStartX = 665;
    const gridStartY = 62;
    const cellW = 24;
    const cellH = 21;
    const gap = 3;

    for (let idx = 0; idx < totalCells; idx += 1) {
      const r = Math.floor(idx / 8);
      const c = idx % 8;
      const cx = gridStartX + c * (cellW + gap);
      const cy = gridStartY + r * (cellH + gap);

      let fill = COLORS.idleCell;
      let stroke = 'none';

      if (idx < activeCutoff) {
        if (idx < usefulCells) {
          if (mode === 'dense') {
            fill = COLORS.dense;
          } else if (idx < c0) {
            fill = COLORS.e0;
          } else if (idx < c0 + c1) {
            fill = COLORS.e1;
          } else {
            fill = COLORS.e2;
          }
        } else if (idx < usefulCells + wasteCells) {
          fill = 'url(#sim-waste-hatch)';
          stroke = '#d14338';
        } else {
          fill = 'url(#sim-stall-hatch)';
        }
      }

      svg.appendChild(
        svgEl('rect', {
          x: cx,
          y: cy,
          width: cellW,
          height: cellH,
          rx: 3,
          fill: fill,
          stroke: stroke,
          'stroke-width': stroke === 'none' ? 0 : 0.7
        })
      );
    }

    // Legend below MXU grid
    const legendY = 286;
    svg.appendChild(
      svgEl('rect', { x: 665, y: legendY, width: 10, height: 10, rx: 2, fill: COLORS.e0 })
    );
    svg.appendChild(
      svgText('Useful: ' + modeData.util + '%', {
        x: 680,
        y: legendY + 9,
        fill: COLORS.ink,
        'font-family': 'JetBrains Mono, monospace',
        'font-size': 9.5,
        'font-weight': 600
      })
    );

    svg.appendChild(
      svgEl('rect', {
        x: 778,
        y: legendY,
        width: 10,
        height: 10,
        rx: 2,
        fill: 'url(#sim-waste-hatch)',
        stroke: '#d14338',
        'stroke-width': 0.7
      })
    );
    svg.appendChild(
      svgText('0-Pad: ' + modeData.wastePct + '%', {
        x: 793,
        y: legendY + 9,
        fill: COLORS.ink,
        'font-family': 'JetBrains Mono, monospace',
        'font-size': 9.5,
        'font-weight': 600
      })
    );

    svg.appendChild(
      svgEl('rect', {
        x: 665,
        y: legendY + 18,
        width: 10,
        height: 10,
        rx: 2,
        fill: 'url(#sim-stall-hatch)'
      })
    );
    svg.appendChild(
      svgText('Idle / Launch Bubble: ' + modeData.idlePct + '%', {
        x: 680,
        y: legendY + 27,
        fill: COLORS.muted,
        'font-family': 'JetBrains Mono, monospace',
        'font-size': 9.5
      })
    );
  }

  function updateCopyAndMetrics(m) {
    const { e0, e1, e2, mode } = state;
    const total = m.total;

    const headingEl = document.getElementById('sim-heading');
    const tagEl = document.getElementById('mode-status-tag');
    const descEl = document.getElementById('mode-banner-desc');
    const shapeEl = document.getElementById('mode-shape-callout');

    const launchesEl = document.getElementById('metric-launches');
    const launchesSubEl = document.getElementById('metric-launches-sub');
    const rowsEl = document.getElementById('metric-rows');
    const rowsSubEl = document.getElementById('metric-rows-sub');
    const wasteEl = document.getElementById('metric-waste');
    const wasteSubEl = document.getElementById('metric-waste-sub');
    const mxuEl = document.getElementById('metric-mxu');
    const mxuSubEl = document.getElementById('metric-mxu-sub');

    if (mode === 'dense') {
      headingEl.textContent = 'Dense Model Baseline (e.g., Llama 3.2 1B)';
      tagEl.textContent = '1 Uniform Matmul \u00b7 Saturated MXU';
      tagEl.className = 'status-tag tag-good';
      descEl.textContent =
        'Every token in the batch goes through the exact same Feed-Forward Network (FFN) weight matrix—one large, uniform matrix multiplication that keeps the TPU MXUs saturated.';
      shapeEl.textContent =
        'Tokens: [' + total + ', Hidden]\nWeights: [Hidden, Intermediate]\n1 shared FFN (no router)';

      launchesEl.textContent = '1';
      launchesSubEl.textContent = 'Single dense matmul';
      rowsEl.textContent = total.toLocaleString();
      rowsSubEl.textContent = total + ' real \u00b7 0 padded zeros';
      wasteEl.textContent = '0.0%';
      wasteSubEl.textContent = 'No router capacity padding';
      mxuEl.textContent = m.dense.util + '%';
      mxuSubEl.textContent = 'Continuous systolic saturation';
    } else if (mode === 'loop') {
      headingEl.textContent = 'MoE Naive Approach #1: For-Loop Over Experts';
      tagEl.textContent = '3 Separate Launches \u00b7 Hardware Starved';
      tagEl.className = 'status-tag tag-bad';
      descEl.textContent =
        'Launches separate small matmuls one expert at a time. Kernel launch overhead creates idle bubbles, and small groups (like ' +
        m.minExpert +
        ' tokens on a 128-row MXU tile) leave ' +
        m.minTileIdlePct +
        '% of that tile idle.';
      shapeEl.textContent =
        'Call 0: [' +
        e0 +
        ', K] @ W_0\nCall 1: [' +
        e1 +
        ', K] @ W_1\nCall 2: [' +
        e2 +
        ', K] @ W_2';

      launchesEl.textContent = '3 (N)';
      launchesSubEl.textContent = 'Scales linearly with experts!';
      rowsEl.textContent = total.toLocaleString();
      rowsSubEl.textContent = 'Fragmented across N launches';
      wasteEl.textContent = '0.0%';
      wasteSubEl.textContent = 'Starved by launch & tile gaps instead';
      mxuEl.textContent = m.loop.util + '%';
      mxuSubEl.textContent = 'Tanks further at 64–128 experts';
    } else if (mode === 'padded') {
      const wasteStr = m.paddedWastePct.toFixed(1) + '%';
      headingEl.textContent = 'MoE Naive Approach #2: Padding All Experts to Max Size';
      tagEl.textContent = wasteStr + ' Wasted FLOPs & HBM Bandwidth';
      tagEl.className = 'status-tag tag-warn';
      descEl.textContent =
        'Pads every expert up to max_tokens = ' +
        m.maxE +
        ' so a single batched matmul can run—forcing the TPU to read and multiply ' +
        m.zeroRows.toLocaleString() +
        ' padded zero rows.';
      shapeEl.textContent =
        'LHS: [3, ' +
        m.maxE +
        ', K] (' +
        m.paddedRows +
        ' rows)\nRHS: [3, K, N]\nWasted zeros: ' +
        m.zeroRows +
        ' rows';

      launchesEl.textContent = '1';
      launchesSubEl.textContent = 'Batched padded matmul';
      rowsEl.textContent = m.paddedRows.toLocaleString();
      rowsSubEl.textContent = total + ' real + ' + m.zeroRows + ' padded zeros';
      wasteEl.textContent = wasteStr;
      wasteSubEl.textContent = m.zeroRows + ' fake token rows multiplied';
      mxuEl.textContent = m.padded.util + '%';
      mxuSubEl.textContent = 'Busy multiplying 0.0 \u00d7 Weights';
    } else {
      headingEl.textContent = 'The Grouped Matmul (gmm / ragged_dot) Solution';
      tagEl.textContent = '1 Fused Launch \u00b7 0% Padding Waste';
      tagEl.className = 'status-tag tag-good';
      descEl.textContent =
        'Packs all ' +
        total +
        ' routed tokens contiguously alongside stacked expert weights [3, K, N] and group_sizes = [' +
        e0 +
        ', ' +
        e1 +
        ', ' +
        e2 +
        '], executing all ragged matmuls in one fused Pallas kernel.';
      shapeEl.textContent =
        'LHS: [' +
        total +
        ', K] (contiguous)\nRHS: [3, K, N] (stacked experts)\ngroup_sizes: [' +
        e0 +
        ', ' +
        e1 +
        ', ' +
        e2 +
        ']';

      launchesEl.textContent = '1';
      launchesSubEl.textContent = 'Single fused Pallas call';
      rowsEl.textContent = total.toLocaleString();
      rowsSubEl.textContent = total + ' real \u00b7 0 padded zeros';
      wasteEl.textContent = '0.0%';
      wasteSubEl.textContent = 'Zero HBM bandwidth wasted';
      mxuEl.textContent = m.gmm.util + '%';
      mxuSubEl.textContent = 'VMEM-tiled systolic saturation';
    }

    document.getElementById('label-e0-count').textContent = e0 + ' tokens';
    document.getElementById('label-e1-count').textContent = e1 + ' tokens';
    document.getElementById('label-e2-count').textContent = e2 + ' tokens';
  }

  function render() {
    const svg = document.getElementById('hardware-simulator-svg');
    if (!svg) {
      return;
    }

    const m = computeMetrics(state.e0, state.e1, state.e2);
    updateCopyAndMetrics(m);

    svg.replaceChildren();
    svg.appendChild(buildDefs());
    renderMemoryPanel(svg, m);
    renderTimelinePanel(svg, m);
    renderMxuPanel(svg, m);
  }

  function triggerReplayAnimation() {
    if (state.rafId !== null) {
      window.cancelAnimationFrame(state.rafId);
      state.rafId = null;
    }

    const durationMs = 950;
    const startTs = window.performance.now();
    state.progress = 0.04;

    function tick(now) {
      const elapsed = now - startTs;
      const t = Math.min(1, elapsed / durationMs);
      // Smooth ease-out cubic
      state.progress = 1 - Math.pow(1 - t, 3);
      render();
      if (t < 1) {
        state.rafId = window.requestAnimationFrame(tick);
      } else {
        state.progress = 1;
        state.rafId = null;
        render();
      }
    }

    state.rafId = window.requestAnimationFrame(tick);
  }

  function setMode(nextMode) {
    const allowed = ['dense', 'loop', 'padded', 'gmm'];
    if (allowed.indexOf(nextMode) === -1) {
      return;
    }
    state.mode = nextMode;

    const tabs = document.querySelectorAll('.mode-tab');
    for (let i = 0; i < tabs.length; i += 1) {
      const tab = tabs[i];
      const active = tab.getAttribute('data-mode') === nextMode;
      tab.classList.toggle('is-active', active);
      tab.setAttribute('aria-selected', active ? 'true' : 'false');
    }

    triggerReplayAnimation();
  }

  function initEvents() {
    const tabs = document.querySelectorAll('.mode-tab');
    for (let i = 0; i < tabs.length; i += 1) {
      tabs[i].addEventListener('click', function (ev) {
        const targetMode = ev.currentTarget.getAttribute('data-mode');
        setMode(targetMode);
      });
    }

    const sliderE0 = document.getElementById('slider-e0');
    const sliderE1 = document.getElementById('slider-e1');
    const sliderE2 = document.getElementById('slider-e2');

    function onSliderInput() {
      state.e0 = clampInt(sliderE0.value, 10, 800, 510);
      state.e1 = clampInt(sliderE1.value, 5, 800, 35);
      state.e2 = clampInt(sliderE2.value, 10, 800, 290);
      state.progress = 1;
      render();
    }

    sliderE0.addEventListener('input', onSliderInput);
    sliderE1.addEventListener('input', onSliderInput);
    sliderE2.addEventListener('input', onSliderInput);

    const presets = document.querySelectorAll('.preset-btn');
    for (let i = 0; i < presets.length; i += 1) {
      presets[i].addEventListener('click', function (ev) {
        const btn = ev.currentTarget;
        state.e0 = clampInt(btn.getAttribute('data-e0'), 10, 800, 510);
        state.e1 = clampInt(btn.getAttribute('data-e1'), 5, 800, 35);
        state.e2 = clampInt(btn.getAttribute('data-e2'), 10, 800, 290);
        sliderE0.value = String(state.e0);
        sliderE1.value = String(state.e1);
        sliderE2.value = String(state.e2);
        triggerReplayAnimation();
      });
    }

    const replayBtn = document.getElementById('btn-replay-step');
    replayBtn.addEventListener('click', function () {
      triggerReplayAnimation();
    });
  }

  document.addEventListener('DOMContentLoaded', function () {
    initEvents();
    render();
  });
})();
