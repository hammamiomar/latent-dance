/**
 * Notifications - Win95 styled toast notifications
 *
 * Displays errors, warnings, and info messages in the corner.
 */

import { motion, AnimatePresence } from 'motion/react';
import { useNotificationStore, type Notification } from '../stores/useNotificationStore';

const TYPE_STYLES: Record<Notification['type'], { bg: string; border: string; icon: string }> = {
  error: {
    bg: 'var(--color-error)',
    border: 'var(--color-error)',
    icon: '✕',
  },
  warning: {
    bg: 'var(--color-warning, #b89000)',
    border: 'var(--color-warning, #b89000)',
    icon: '⚠',
  },
  success: {
    bg: 'var(--color-accent)',
    border: 'var(--color-accent)',
    icon: '✓',
  },
  info: {
    bg: 'var(--color-text-muted)',
    border: 'var(--color-text-muted)',
    icon: 'i',
  },
};

function NotificationItem({ notification }: { notification: Notification }) {
  const remove = useNotificationStore((s) => s.remove);
  const style = TYPE_STYLES[notification.type];

  return (
    <motion.div
      layout
      initial={{ opacity: 0, x: 50, scale: 0.9 }}
      animate={{ opacity: 1, x: 0, scale: 1 }}
      exit={{ opacity: 0, x: 50, scale: 0.9 }}
      transition={{ type: 'spring', stiffness: 500, damping: 30 }}
      className="win95-panel relative"
      style={{
        borderColor: style.border,
        maxWidth: 300,
      }}
    >
      {/* Title bar */}
      <div
        className="flex items-center gap-2 px-2 py-1"
        style={{ background: style.bg }}
      >
        <span className="text-xs font-bold" style={{ color: 'var(--color-void-deep)' }}>
          {style.icon}
        </span>
        <span
          className="text-xs font-bold uppercase flex-1"
          style={{ color: 'var(--color-void-deep)' }}
        >
          {notification.type}
        </span>
        <button
          onClick={() => remove(notification.id)}
          className="text-xs font-bold hover:opacity-70"
          style={{ color: 'var(--color-void-deep)' }}
        >
          ×
        </button>
      </div>

      {/* Message */}
      <div className="p-2">
        <p className="text-xs" style={{ color: 'var(--color-text-primary)' }}>
          {notification.message}
        </p>
      </div>
    </motion.div>
  );
}

export function Notifications() {
  const notifications = useNotificationStore((s) => s.notifications);

  return (
    <div className="absolute bottom-4 right-4 z-[200] flex flex-col gap-2">
      <AnimatePresence mode="popLayout">
        {notifications.map((notification) => (
          <NotificationItem key={notification.id} notification={notification} />
        ))}
      </AnimatePresence>
    </div>
  );
}
