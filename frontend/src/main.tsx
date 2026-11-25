import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { App } from "./App";
import "./styles.css";
import { keepPicks } from "./state/picks";
import { loadPlugins } from "./plugins/host";

keepPicks();
void loadPlugins();

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
