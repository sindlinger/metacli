import path from 'path';
import fs from 'fs-extra';
import { fileURLToPath } from 'url';

export interface ProjectDefaults {
  symbol?: string;
  period?: string;
  subwindow?: number;
  indicator?: string | null;
  portable?: boolean;
  profile?: string | null;
}

export interface ProjectInfo {
  project: string;
  libs: string;
  terminal?: string;
  metaeditor?: string;
  data_dir?: string;
  updated_at?: string;
  created_at?: string;
  defaults?: ProjectDefaults;
  [key: string]: unknown;
}

export interface ProjectsFile {
  last_project?: string;
  projects: Record<string, ProjectInfo>;
}

const ROOT_DIR = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const PROJECTS_FILE = path.join(ROOT_DIR, '..', 'mtcli_projects.json');

export class ProjectStore {
  async load(): Promise<ProjectsFile> {
    if (!(await fs.pathExists(PROJECTS_FILE))) {
      return { projects: {} };
    }
    const data = await fs.readFile(PROJECTS_FILE, 'utf8');
    if (!data.trim()) {
      return { projects: {} };
    }
    return JSON.parse(data) as ProjectsFile;
  }

  async save(content: ProjectsFile): Promise<void> {
    await fs.writeFile(PROJECTS_FILE, JSON.stringify(content, null, 2));
  }

  async useOrThrow(id?: string): Promise<ProjectInfo> {
    const file = await this.load();
    const candidate = id || file.last_project;
    if (!candidate) {
      throw new Error('Nenhum projeto ativo. Use "mtcli project save --id <nome> ..." ou --project.');
    }
    const info = file.projects[candidate];
    if (!info) {
      throw new Error(`Projeto "${candidate}" não registrado em mtcli_projects.json.`);
    }
    return info;
  }

  async setProject(id: string, update: Partial<ProjectInfo>, setDefault = false): Promise<ProjectInfo> {
    const file = await this.load();
    const existing = file.projects[id] || { project: id };
    const next: ProjectInfo = {
      ...existing,
      ...update,
      project: id,
      updated_at: new Date().toISOString(),
      created_at: existing.created_at || new Date().toISOString(),
    };
    file.projects[id] = next;
    if (setDefault || !file.last_project) {
      file.last_project = id;
    }
    await this.save(file);
    return next;
  }

  async updateDefaults(id: string, defaults: ProjectDefaults): Promise<ProjectInfo> {
    const file = await this.load();
    const project = file.projects[id];
    if (!project) {
      throw new Error(`Projeto "${id}" não encontrado.`);
    }
    project.defaults = {
      ...project.defaults,
      ...defaults,
    };
    project.updated_at = new Date().toISOString();
    await this.save(file);
    return project;
  }

  async show(): Promise<ProjectsFile> {
    return this.load();
  }
}

export function requireField<T>(value: T | undefined, message: string): T {
  if (value === undefined || value === null || value === '') {
    throw new Error(message);
  }
  return value;
}

export function repoRoot(): string {
  return path.resolve(ROOT_DIR, '..');
}

export function projectsFilePath(): string {
  return PROJECTS_FILE;
}
