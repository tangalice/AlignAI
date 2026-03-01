import { test, expect } from "@playwright/test";

test.describe("AlignAI app", () => {
  test("loads and shows main UI", async ({ page }) => {
    await page.goto("/");

    await expect(page.getByRole("heading", { name: /AlignAI/i })).toBeVisible();
    await expect(page.getByRole("button", { name: /Exercise Workout/i })).toBeVisible();
    await expect(page.getByRole("button", { name: /AI Generated Video/i })).toBeVisible();
  });

  test("can switch between tabs", async ({ page }) => {
    await page.goto("/");

    await page.getByRole("button", { name: /AI Generated Video/i }).click();
    await expect(page.getByLabel(/Describe the workout/i)).toBeVisible();

    await page.getByRole("button", { name: /Exercise Workout/i }).click();
    await expect(page.getByLabel(/Search workout/i)).toBeVisible();
  });

  test("exercise search input is available on Exercise tab", async ({ page }) => {
    await page.goto("/");
    await expect(page.getByPlaceholder(/Bicycle Crunches/i)).toBeVisible();
  });
});
