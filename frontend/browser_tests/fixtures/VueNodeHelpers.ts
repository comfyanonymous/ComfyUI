/**
 * Vue Node Test Helpers
 */
import type { Locator, Page } from '@playwright/test'

import { TestIds } from './selectors'
import { VueNodeFixture } from './utils/vueNodeFixtures'

export class VueNodeHelpers {
  constructor(private page: Page) {}

  /**
   * Get locator for all Vue node components in the DOM
   */
  get nodes(): Locator {
    return this.page.locator('[data-node-id]')
  }

  /**
   * Get locator for a Vue node by its NodeId
   */
  getNodeLocator(nodeId: string): Locator {
    return this.page.locator(`[data-node-id="${nodeId}"]`)
  }

  /**
   * Get locator for selected Vue node components (using visual selection indicators)
   */
  get selectedNodes(): Locator {
    return this.page.locator('[data-node-id].outline-node-component-outline')
  }

  /**
   * Get locator for a Vue node by the node's title (displayed name in the header)
   */
  getNodeByTitle(title: string): Locator {
    return this.page.locator(`[data-node-id]`).filter({ hasText: title })
  }

  /**
   * Get total count of Vue nodes in the DOM
   */
  async getNodeCount(): Promise<number> {
    return await this.nodes.count()
  }

  /**
   * Get count of selected Vue nodes
   */
  async getSelectedNodeCount(): Promise<number> {
    return await this.selectedNodes.count()
  }

  /**
   * Get all Vue node IDs currently in the DOM
   */
  async getNodeIds(): Promise<string[]> {
    return await this.nodes.evaluateAll((nodes) =>
      nodes
        .map((n) => n.getAttribute('data-node-id'))
        .filter((id): id is string => id !== null)
    )
  }

  /**
   * Select a specific Vue node by ID
   */
  async selectNode(nodeId: string): Promise<void> {
    await this.page
      .locator(`[data-node-id="${nodeId}"] .lg-node-header`)
      .click()
  }

  /**
   * Select multiple Vue nodes by IDs using Ctrl+click
   */
  async selectNodes(nodeIds: string[]): Promise<void> {
    if (nodeIds.length === 0) return

    // Select first node normally
    await this.selectNode(nodeIds[0])

    // Add additional nodes with Ctrl+click on header
    for (let i = 1; i < nodeIds.length; i++) {
      await this.page
        .locator(`[data-node-id="${nodeIds[i]}"] .lg-node-header`)
        .click({
          modifiers: ['Control']
        })
    }
  }

  /**
   * Clear all selections by clicking empty space
   */
  async clearSelection(): Promise<void> {
    await this.page.mouse.click(50, 50)
  }

  /**
   * Delete selected Vue nodes using Delete key
   */
  async deleteSelected(): Promise<void> {
    await this.page.locator('#graph-canvas').focus()
    await this.page.keyboard.press('Delete')
  }

  /**
   * Delete selected Vue nodes using Backspace key
   */
  async deleteSelectedWithBackspace(): Promise<void> {
    await this.page.locator('#graph-canvas').focus()
    await this.page.keyboard.press('Backspace')
  }

  /**
   * Return a DOM-focused VueNodeFixture for the first node matching the title.
   * Resolves the node id up front so subsequent interactions survive title changes.
   */
  async getFixtureByTitle(title: string): Promise<VueNodeFixture> {
    const node = this.getNodeByTitle(title).first()
    await node.waitFor({ state: 'visible' })

    const nodeId = await node.evaluate((el) => el.getAttribute('data-node-id'))
    if (!nodeId) {
      throw new Error(
        `Vue node titled "${title}" is missing its data-node-id attribute`
      )
    }

    return new VueNodeFixture(this.getNodeLocator(nodeId))
  }

  /**
   * Wait for Vue nodes to be rendered
   */
  async waitForNodes(expectedCount?: number): Promise<void> {
    if (expectedCount !== undefined) {
      await this.page.waitForFunction(
        (count) => document.querySelectorAll('[data-node-id]').length >= count,
        expectedCount
      )
    } else {
      await this.page.waitForSelector('[data-node-id]')
    }
  }

  /**
   * Get a specific widget by node title and widget name
   */
  getWidgetByName(nodeTitle: string, widgetName: string): Locator {
    return this.getNodeByTitle(nodeTitle).getByLabel(widgetName, {
      exact: true
    })
  }

  /**
   * Get controls for input number widgets (increment/decrement buttons and input)
   */
  getInputNumberControls(widget: Locator) {
    return {
      input: widget.locator('input'),
      decrementButton: widget.getByTestId(TestIds.widgets.decrement),
      incrementButton: widget.getByTestId(TestIds.widgets.increment)
    }
  }

  /**
   * Enter the subgraph of a node.
   * @param nodeId - The ID of the node to enter the subgraph of. If not provided, the first matched subgraph will be entered.
   */
  async enterSubgraph(nodeId?: string): Promise<void> {
    const locator = nodeId ? this.getNodeLocator(nodeId) : this.page
    const editButton = locator.getByTestId(TestIds.widgets.subgraphEnterButton)
    await editButton.click()
  }
}
