---
name: Equinox Next
description: Evidence-first research operations with drafting-instrument precision
colors:
  accent-blue: "#2F6F91"
  accent-blue-strong: "#245A78"
  paper: "#F4F6F7"
  surface: "#FFFFFF"
  surface-step: "#E9EEF1"
  ink: "#17242C"
  ink-muted: "#566873"
  rule: "#C8D2D8"
  success: "#237A57"
  warning: "#A26113"
  danger: "#A63D40"
typography:
  headline:
    fontFamily: "ui-sans-serif, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif"
    fontSize: "clamp(1.5rem, 2vw, 2.25rem)"
    fontWeight: 650
    lineHeight: 1.1
    letterSpacing: "-0.02em"
  body:
    fontFamily: "ui-sans-serif, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif"
    fontSize: "0.9375rem"
    fontWeight: 400
    lineHeight: 1.5
    letterSpacing: "normal"
  label:
    fontFamily: "ui-monospace, SFMono-Regular, Consolas, monospace"
    fontSize: "0.75rem"
    fontWeight: 600
    lineHeight: 1.35
    letterSpacing: "0.02em"
rounded:
  control: "4px"
  panel: "6px"
spacing:
  xs: "4px"
  sm: "8px"
  md: "16px"
  lg: "24px"
  xl: "32px"
components:
  button-primary:
    backgroundColor: "{colors.accent-blue}"
    textColor: "{colors.surface}"
    rounded: "{rounded.control}"
    padding: "10px 14px"
  button-secondary:
    backgroundColor: "{colors.surface}"
    textColor: "{colors.ink}"
    rounded: "{rounded.control}"
    padding: "9px 13px"
  field:
    backgroundColor: "{colors.surface}"
    textColor: "{colors.ink}"
    rounded: "{rounded.control}"
    padding: "10px 12px"
---

# Design System: Equinox Next

## Overview

**Creative North Star: "The Verification Drafting Table"**

Equinox feels like a digital drafting table prepared for a research review: calibrated,
ordered, and dense with useful evidence. Alignment rules, coordinate-like labels, proof
panes, and synchronized views make lineage legible without dressing the product as a
generic developer console.

The interface is calm but not sparse. Operators should see enough state to decide whether
work is valid, stalled, expensive, or inadmissible. The visual system never promotes a
model assessment above deterministic evidence and never turns a failure class into a
decorative status color.

**Key characteristics:**

- Flat, rule-defined layers with no drop shadows.
- Compact operating density and generous separation between distinct tasks.
- One oxide-blue product accent; semantic colors remain separate.
- Full-width working canvases paired with exact outline and evidence panes.
- System sans for reading; mono only for identifiers, digests, units, and logs.

## Colors

The palette uses cool drafting-paper neutrals and a measured oxide blue so dense evidence
remains calm in both light and dark operating environments.

### Primary

- **Oxide blue** (`#2F6F91`): primary actions, selected paths, links, and active focus.
- **Deep oxide** (`#245A78`): hover and active states.

### Neutral

- **Drafting paper** (`#F4F6F7`): application background.
- **Instrument white** (`#FFFFFF`): primary work surfaces.
- **Blue-gray step** (`#E9EEF1`): secondary surfaces, table headers, and inactive tracks.
- **Carbon ink** (`#17242C`): primary text and decisive geometry.
- **Graphite note** (`#566873`): secondary text and metadata.
- **Calibration rule** (`#C8D2D8`): borders, graph edges, dividers, and plot axes.

### Semantic

- **Accepted green** (`#237A57`): accepted scientific facts and healthy completion.
- **Attention amber** (`#A26113`): retry, partial, degraded, and cleanup warning.
- **Integrity red** (`#A63D40`): failure, exclusion, and integrity violation.

**The one-accent rule.** Oxide blue identifies product interaction and selection. Green,
amber, and red communicate outcomes only; they never substitute for hierarchy.

## Typography

**Display Font:** system UI sans
**Body Font:** system UI sans
**Label/Mono Font:** system monospace

**Character:** Workmanlike UI type supports rapid scanning and reliable rendering inside
the Docker-built dashboard. Monospace is reserved for machine-shaped content rather than
used as a technical costume.

### Hierarchy

- **Headline** (650, `clamp(1.5rem, 2vw, 2.25rem)`, 1.1): route title and selected object.
- **Section title** (650, `1rem`, 1.3): major working regions.
- **Body** (400, `0.9375rem`, 1.5): explanations, states, and operational prose.
- **Compact body** (400, `0.8125rem`, 1.45): tables and evidence metadata.
- **Machine label** (600, `0.75rem`, 0.02em): IDs, digests, units, versions, and logs.

**The readable-evidence rule.** Prose stays proportional. A digest may be mono; the
sentence explaining why it matters is not.

## Layout

The desktop shell uses a 224px navigation rail and a fluid work area. Wide proof routes
use a three-region composition: graph or outline, synchronized visual evidence, and
verification detail. The run list and iteration views use native tables with sticky
headers rather than grids of cards.

The spacing unit is 4px, with 8/16/24/32px steps. Related controls stay within 8px;
panels use 16px internal padding; separate decisions receive at least 24px. At 1100px,
evidence panes stack below the primary graph. At 760px, the rail becomes a compact top
navigation and tables retain horizontal overflow with pinned first columns.

## Elevation & Depth

The system has no drop shadows. Depth comes from one-pixel calibration rules, subtle
background steps, selected-state fills, typography, and spatial separation. Dialogs are
reserved for protected focus and use an opaque backdrop, not glass.

**The flat-authority rule.** A border or a tonal step may define a surface. Never add a
shadow to make an otherwise unclear hierarchy legible.

## Shapes

Controls use a precise 4px radius. Panels use 6px. Status lozenges may use a compact pill
shape because they are small categorical tokens; action buttons and work surfaces never
become pills. Graph nodes use clipped-corner or square drafting silhouettes expressed
with borders, not illustrated sketch effects.

## Components

### Buttons

- **Shape:** compact rectangle with 4px corners.
- **Primary:** oxide blue with white text; one primary action per local task region.
- **Secondary:** surface background, one-pixel rule, carbon text.
- **Hover / Focus:** deep oxide on hover; a 2px oxide focus ring with 2px offset.
- **Disabled:** step background and muted text; no pointer affordance.

### Status labels

- **Style:** semantic text plus a pale hue-biased fill and one-pixel border.
- **Language:** names the outcome class: accepted, candidate failed, abstained, retrying,
  infrastructure failed, or integrity violation.

### Tables

- **Style:** full-width native tables with sticky tonal headers, row rules, and
  click-through rows only when the row performs navigation.
- **Selection:** pale oxide fill, oxide text, and a visible focus outline.

### Cards / containers

- **Corner style:** 6px only where a bounded panel is necessary.
- **Background:** instrument white or one neutral step.
- **Shadow strategy:** none.
- **Border:** one calibration rule.
- **Internal padding:** 16px, reduced to 12px for dense inspector sections.

### Inputs / fields

- **Style:** instrument-white fill, one-pixel rule, 4px radius.
- **Focus:** oxide border and 2px focus ring.
- **Error / disabled:** error text names the problem and recovery; disabled state keeps
  its label readable.

### Navigation

The persistent rail uses route names and compact line icons. Active routes receive a pale
oxide background and carbon text; inactive items remain plain. Navigation does not imitate
a dashboard metric surface.

### Evidence comparator

Reference, source, candidate, and sibling renders share one synchronized viewport frame.
The selected evidence role is expressed through a tab and URL state. Metadata remains
adjacent and visible; hidden evidence has a designed locked state rather than disappearing.

### Verification graph

Canvas nodes expose step type, outcome, attempts, and cache state. Selection synchronizes
with the inspector and URL. A table outline with dependencies, status, and evidence roles
is always adjacent or one tab away.

## Do's and Don'ts

### Do:

- **Do** lead each route with the current decision and the next valid action.
- **Do** keep deterministic evidence and model assessments visually and verbally distinct.
- **Do** expose IDs, digests, versions, attempts, uncertainty, and exclusion reasons.
- **Do** make graph selection, filters, and inspector state addressable in the URL.
- **Do** design empty, loading, error, partial, degraded, and hidden-evidence states.

### Don't:

- **Don't** use shadows, glass, gradients, decorative grids, or filler metrics.
- **Don't** use a clickable visual treatment on static information.
- **Don't** collapse retry, candidate failure, abstention, disagreement, and integrity
  violation into one red error state.
- **Don't** present mock judge output as objective quality or calibrated human agreement.
- **Don't** use monospace for ordinary body copy.
