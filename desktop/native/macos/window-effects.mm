/**
 * Native macOS window effects for hambajuba2ba.
 *
 * Compiled to libMacWindowEffects.dylib via scripts/build-native.sh.
 * Loaded by Bun FFI in src/bun/index.ts. All functions take the raw
 * NSWindow* pointer from Electrobun's BrowserWindow.ptr.
 *
 * Five effects:
 *   1. Window shadow — grounded object on desktop
 *   2. Traffic light positioning — close/minimize/zoom in the body
 *   3. Native drag region — smooth window dragging with Space transitions
 *   4. Subdued traffic lights — gray by default, colored on hover
 *   5. Content aspect ratio — OS-enforced resize constraint (no flicker)
 */

#import <Cocoa/Cocoa.h>
#import <QuartzCore/QuartzCore.h>

// ---------------------------------------------------------------------------
// Tracking view — wraps traffic light container for mouse enter/exit
// ---------------------------------------------------------------------------

static NSString *const kTrackingViewId = @"HambaTrafficLightTracker";

@interface HambaTrafficLightTracker : NSView
@property (nonatomic, weak) NSView *buttonContainer;
@end

@implementation HambaTrafficLightTracker

- (void)updateTrackingAreas {
    [super updateTrackingAreas];
    // Remove old tracking areas
    for (NSTrackingArea *area in [self trackingAreas]) {
        [self removeTrackingArea:area];
    }
    NSTrackingArea *ta = [[NSTrackingArea alloc]
        initWithRect:[self bounds]
             options:(NSTrackingMouseEnteredAndExited |
                      NSTrackingActiveAlways |
                      NSTrackingInVisibleRect)
               owner:self
            userInfo:nil];
    [self addTrackingArea:ta];
}

- (void)mouseEntered:(NSEvent *)event {
    (void)event;
    if (self.buttonContainer) {
        [NSAnimationContext runAnimationGroup:^(NSAnimationContext *ctx) {
            ctx.duration = 0.15;
            self.buttonContainer.animator.alphaValue = 1.0;
        }];
    }
}

- (void)mouseExited:(NSEvent *)event {
    (void)event;
    if (self.buttonContainer) {
        [NSAnimationContext runAnimationGroup:^(NSAnimationContext *ctx) {
            ctx.duration = 0.3;
            self.buttonContainer.animator.alphaValue = 0.4;
        }];
    }
}

- (BOOL)isOpaque { return NO; }
- (void)drawRect:(NSRect)dirtyRect { (void)dirtyRect; }

@end

// ---------------------------------------------------------------------------
// Native drag view — transparent NSView that forwards mouseDown to window drag
// ---------------------------------------------------------------------------

static NSString *const kDragViewId = @"HambajubaDragView";

@interface HambajubaDragView : NSView
@end

@implementation HambajubaDragView

- (BOOL)isOpaque {
    return NO;
}

- (void)drawRect:(NSRect)dirtyRect {
    (void)dirtyRect; // invisible — drawing handled by the webview
}

- (void)mouseDown:(NSEvent *)event {
    NSWindow *window = [self window];
    if (window != nil && event != nil) {
        [window performWindowDragWithEvent:event];
    }
}

@end

// ---------------------------------------------------------------------------
// Helper — find our drag view in the content view's subviews
// ---------------------------------------------------------------------------

static HambajubaDragView *findDragView(NSView *contentView) {
    for (NSView *subview in [contentView subviews]) {
        if ([subview isKindOfClass:[HambajubaDragView class]] &&
            [[subview identifier] isEqualToString:kDragViewId]) {
            return (HambajubaDragView *)subview;
        }
    }
    return nil;
}

// ---------------------------------------------------------------------------
// Public API — called via Bun FFI
// ---------------------------------------------------------------------------

extern "C" bool ensureWindowShadow(void *windowPtr) {
    if (windowPtr == nullptr) return false;

    __block BOOL success = NO;
    dispatch_sync(dispatch_get_main_queue(), ^{
        NSWindow *window = (__bridge NSWindow *)windowPtr;
        if (![window isKindOfClass:[NSWindow class]]) return;

        [window setHasShadow:YES];
        [window invalidateShadow];
        success = YES;
    });

    return success;
}

extern "C" bool setWindowTrafficLightsPosition(void *windowPtr,
                                                double x,
                                                double yFromTop) {
    if (windowPtr == nullptr) return false;

    __block BOOL success = NO;
    dispatch_sync(dispatch_get_main_queue(), ^{
        NSWindow *window = (__bridge NSWindow *)windowPtr;
        if (![window isKindOfClass:[NSWindow class]]) return;

        NSButton *close    = [window standardWindowButton:NSWindowCloseButton];
        NSButton *minimize = [window standardWindowButton:NSWindowMiniaturizeButton];
        NSButton *zoom     = [window standardWindowButton:NSWindowZoomButton];

        if (close == nil || minimize == nil || zoom == nil) return;

        NSView *container = [close superview];
        if (container == nil) return;

        // Spacing between buttons (typically ~20px)
        CGFloat spacing = NSMinX(minimize.frame) - NSMinX(close.frame);
        if (spacing <= 0) {
            spacing = close.frame.size.width + 6.0;
        }

        // Convert yFromTop to the container's coordinate system
        BOOL flipped = [container isFlipped];
        CGFloat targetY = yFromTop;
        if (!flipped) {
            targetY = container.frame.size.height - yFromTop
                    - close.frame.size.height;
        }
        targetY = MAX(0.0, targetY);

        // Position each button
        CGFloat currentX = x;
        NSArray<NSButton *> *buttons = @[close, minimize, zoom];
        for (NSButton *button in buttons) {
            [button setFrameOrigin:NSMakePoint(currentX, targetY)];
            currentX += spacing;
        }

        [container setNeedsLayout:YES];
        [container layoutSubtreeIfNeeded];
        [window invalidateShadow];
        success = YES;
    });

    return success;
}

extern "C" bool setNativeWindowDragRegion(void *windowPtr,
                                           double x,
                                           double height) {
    if (windowPtr == nullptr) return false;

    __block BOOL success = NO;
    dispatch_sync(dispatch_get_main_queue(), ^{
        NSWindow *window = (__bridge NSWindow *)windowPtr;
        if (![window isKindOfClass:[NSWindow class]]) return;

        NSView *contentView = [window contentView];
        if (contentView == nil) return;

        CGFloat dragX      = MAX(0.0, x);
        CGFloat dragHeight = MAX(0.0, height);
        CGFloat dragWidth  = MAX(0.0, contentView.bounds.size.width - dragX);
        if (dragHeight <= 0.0 || dragWidth <= 0.0) return;

        // Content view may be flipped (y=0 at top) or not (y=0 at bottom)
        BOOL flipped = [contentView isFlipped];
        CGFloat dragY = flipped
            ? 0.0
            : contentView.bounds.size.height - dragHeight;
        dragY = MAX(0.0, dragY);

        HambajubaDragView *dragView = findDragView(contentView);
        if (dragView == nil) {
            dragView = [[HambajubaDragView alloc] initWithFrame:NSZeroRect];
            [dragView setIdentifier:kDragViewId];
        }

        [dragView setFrame:NSMakeRect(dragX, dragY, dragWidth, dragHeight)];
        [dragView setAutoresizingMask:NSViewWidthSizable];

        if ([dragView superview] == nil) {
            [contentView addSubview:dragView
                         positioned:NSWindowAbove
                         relativeTo:nil];
        }

        success = YES;
    });

    return success;
}

extern "C" bool setTrafficLightsSubdued(void *windowPtr) {
    if (windowPtr == nullptr) return false;

    __block BOOL success = NO;
    dispatch_sync(dispatch_get_main_queue(), ^{
        NSWindow *window = (__bridge NSWindow *)windowPtr;
        if (![window isKindOfClass:[NSWindow class]]) return;

        NSButton *close = [window standardWindowButton:NSWindowCloseButton];
        if (close == nil) return;

        NSView *container = [close superview];
        if (container == nil) return;

        // Start subdued
        container.alphaValue = 0.4;

        // Add tracking view over the button container for hover detection
        NSView *contentView = [window contentView];
        if (contentView == nil) return;

        // Remove existing tracker if any
        for (NSView *sub in [contentView subviews]) {
            if ([[sub identifier] isEqualToString:kTrackingViewId]) {
                [sub removeFromSuperview];
                break;
            }
        }

        HambaTrafficLightTracker *tracker = [[HambaTrafficLightTracker alloc]
            initWithFrame:container.frame];
        [tracker setIdentifier:kTrackingViewId];
        tracker.buttonContainer = container;
        [tracker setAutoresizingMask:NSViewMaxXMargin | NSViewMinYMargin];

        [contentView addSubview:tracker
                     positioned:NSWindowAbove
                     relativeTo:nil];

        success = YES;
    });

    return success;
}

extern "C" bool setWindowAspectRatio(void *windowPtr,
                                      double width,
                                      double height) {
    if (windowPtr == nullptr) return false;

    __block BOOL success = NO;
    dispatch_sync(dispatch_get_main_queue(), ^{
        NSWindow *window = (__bridge NSWindow *)windowPtr;
        if (![window isKindOfClass:[NSWindow class]]) return;

        [window setContentAspectRatio:NSMakeSize(width, height)];
        [window setMinSize:NSMakeSize(width, height)];
        success = YES;
    });

    return success;
}
