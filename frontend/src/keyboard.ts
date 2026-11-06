/** True when a key event comes from a text field or menu (global shortcuts
 * should then leave it alone). */
export const isTyping = (e: Event) =>
  e.target instanceof Element && !!e.target.closest("input, textarea, select");
