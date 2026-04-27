/**
 * Notification Store - Simple toast notifications
 *
 * Provides user feedback for errors, warnings, and success messages.
 */

import { create } from 'zustand';

export type NotificationType = 'error' | 'warning' | 'success' | 'info';

export interface Notification {
  id: string;
  type: NotificationType;
  message: string;
  duration?: number; // ms, 0 = persistent
}

interface NotificationState {
  notifications: Notification[];
  add: (type: NotificationType, message: string, duration?: number) => string;
  remove: (id: string) => void;
  clear: () => void;
}

let notificationId = 0;

export const useNotificationStore = create<NotificationState>((set, get) => ({
  notifications: [],

  add: (type, message, duration = 4000) => {
    const id = `notification-${++notificationId}`;
    const notification: Notification = { id, type, message, duration };

    set((state) => ({
      notifications: [...state.notifications, notification],
    }));

    // Auto-remove after duration (unless persistent)
    if (duration > 0) {
      setTimeout(() => {
        get().remove(id);
      }, duration);
    }

    return id;
  },

  remove: (id) => {
    set((state) => ({
      notifications: state.notifications.filter((n) => n.id !== id),
    }));
  },

  clear: () => {
    set({ notifications: [] });
  },
}));

// Convenience functions for direct access
export const notify = {
  error: (message: string, duration?: number) =>
    useNotificationStore.getState().add('error', message, duration),
  warning: (message: string, duration?: number) =>
    useNotificationStore.getState().add('warning', message, duration),
  success: (message: string, duration?: number) =>
    useNotificationStore.getState().add('success', message, duration),
  info: (message: string, duration?: number) =>
    useNotificationStore.getState().add('info', message, duration),
};
