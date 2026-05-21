// web/src/main.ts
import { app } from "../../../scripts/app.js";

// web/src/api.ts
var SETTINGS_ROUTE = "/simple-syrup/settings";
async function getSettings(fetchImpl = fetch) {
  const response = await fetchImpl(SETTINGS_ROUTE);
  if (!response.ok) {
    throw new Error(
      `Could not load SimpleSyrup settings. Backend returned ${String(response.status)}.`
    );
  }
  return parseSettings(await response.json());
}
async function saveSettings(settings, fetchImpl = fetch) {
  const response = await fetchImpl(SETTINGS_ROUTE, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(settings)
  });
  if (!response.ok) {
    throw new Error(
      `Could not save SimpleSyrup settings. Backend returned ${String(response.status)}.`
    );
  }
  return parseSettings(await response.json());
}
function parseSettings(payload) {
  if (!isSettingsPayload(payload)) {
    throw new Error(
      "SimpleSyrup settings payload is invalid. Expected show_downloadable_models to be a boolean."
    );
  }
  return {
    show_downloadable_models: payload.show_downloadable_models
  };
}
function isSettingsPayload(payload) {
  return typeof payload === "object" && payload !== null && typeof payload.show_downloadable_models === "boolean";
}

// web/src/settings.ts
var SIMPLE_SYRUP_SETTING_ID = "SimpleSyrup.ShowDownloadableModels";
var SIMPLE_SYRUP_SETTING_LABEL = "SimpleSyrup: Show downloadable models in loader dropdowns";
var SIMPLE_SYRUP_SETTING_DESCRIPTION = "Show known downloadable SAM, GroundingDINO, and ViTMatte models even when they are not installed locally.";
var DEFAULT_SETTINGS = {
  show_downloadable_models: true
};
async function registerSimpleSyrupSettings(app2, api = { getSettings, saveSettings }, logger = console) {
  let initialSettings = DEFAULT_SETTINGS;
  try {
    initialSettings = await api.getSettings();
  } catch (error) {
    logger.warn(
      "Could not load SimpleSyrup settings. Using the default setting until the backend is available.",
      error
    );
  }
  let savedSettings = initialSettings;
  const setting = app2.ui.settings.addSetting({
    id: SIMPLE_SYRUP_SETTING_ID,
    name: SIMPLE_SYRUP_SETTING_LABEL,
    type: "boolean",
    defaultValue: initialSettings.show_downloadable_models,
    tooltip: SIMPLE_SYRUP_SETTING_DESCRIPTION,
    onChange: async (value) => {
      try {
        const saved = await api.saveSettings({
          show_downloadable_models: value
        });
        savedSettings = saved;
        setting.value = saved.show_downloadable_models;
      } catch (error) {
        logger.warn(
          "Could not save SimpleSyrup settings. The backend rejected the setting update.",
          error
        );
        setting.value = savedSettings.show_downloadable_models;
      }
    }
  });
  setting.value = initialSettings.show_downloadable_models;
}

// web/src/main.ts
var comfyApp = app;
comfyApp.registerExtension({
  name: "SimpleSyrup.Settings",
  async setup(appInstance) {
    await registerSimpleSyrupSettings(appInstance);
  }
});
