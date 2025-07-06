declare module "https://cdn.jsdelivr.net/*" {
  export const loadPyodide: (cfg: any) => Promise<any>;
}
