import { useSelectedDataset } from "../state/datasets";
import { bytesLabel, kindLabel, shapeLabel } from "../format";

export function Inspector() {
  const d = useSelectedDataset();
  if (!d) return <p className="muted">Select a dataset to see its details.</p>;

  return (
    <div className="inspector">
      <dl className="props">
        <dt>Title</dt>
        <dd>{d.title}</dd>
        <dt>Type</dt>
        <dd>
          {kindLabel(d)}
          {d.signal_type && <span className="muted"> · {d.signal_type}</span>}
        </dd>
        <dt>Shape</dt>
        <dd>{shapeLabel(d)}</dd>
        <dt>Data</dt>
        <dd>
          {d.dtype}, {bytesLabel(d.nbytes)}
          {d.lazy && <span className="muted"> · not loaded</span>}
        </dd>
      </dl>

      <h3>Axes</h3>
      <table className="axes">
        <thead>
          <tr>
            <th>Name</th>
            <th>Size</th>
            <th>Scale</th>
            <th>Offset</th>
            <th>Units</th>
          </tr>
        </thead>
        <tbody>
          {d.axes.map((ax, i) => (
            <tr key={i} className={ax.navigate ? "nav-axis" : undefined}>
              <td>{ax.name || <span className="muted">axis {i}</span>}</td>
              <td>{ax.size}</td>
              <td>{ax.scale.toPrecision(4)}</td>
              <td>{ax.offset.toPrecision(4)}</td>
              <td>{ax.units}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
