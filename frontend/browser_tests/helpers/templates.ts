import type { Locator, Page } from '@playwright/test'
import { expect } from '@playwright/test'
import path from 'path'

import type {
  TemplateInfo,
  WorkflowTemplates
} from '../../src/platform/workflow/templates/types/template'
import { TestIds } from '../fixtures/selectors'

export class ComfyTemplates {
  readonly content: Locator
  readonly allTemplateCards: Locator

  constructor(readonly page: Page) {
    this.content = page.getByTestId(TestIds.templates.content)
    this.allTemplateCards = page.locator('[data-testid^="template-workflow-"]')
  }

  async expectMinimumCardCount(count: number) {
    await expect(async () => {
      const cardCount = await this.allTemplateCards.count()
      expect(cardCount).toBeGreaterThanOrEqual(count)
    }).toPass({
      timeout: 1_000
    })
  }

  async loadTemplate(id: string) {
    const templateCard = this.content.getByTestId(
      TestIds.templates.workflowCard(id)
    )
    await templateCard.scrollIntoViewIfNeeded()
    await templateCard.getByRole('img').click()
  }

  async getAllTemplates(): Promise<TemplateInfo[]> {
    const templates: WorkflowTemplates[] = await this.page.evaluate(() =>
      window.app!.api.getCoreWorkflowTemplates()
    )
    return templates.flatMap((t) => t.templates)
  }

  getTemplatePath(filename: string): string {
    return path.join('public', 'templates', filename)
  }
}
