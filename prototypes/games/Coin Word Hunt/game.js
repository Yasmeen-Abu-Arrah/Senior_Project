/* =====================================================
   FIND THE WORD: C-O-I-N
   Top-down adventure using Schwarnhild tileset assets
   ===================================================== */
'use strict';

// ── CONFIG ──────────────────────────────────────────
const TILE  = 32;
const COLS  = 20;
const ROWS  = 15;
const W     = COLS * TILE;   // 640
const H     = ROWS * TILE;   // 480

const ASSETS = '../New folder/basic_tileset_and_assets_standard_v3.0/';

// ── MAP  (0 = grass/walkable, 1 = tree/wall) ────────
//   Letters:  C=top-left  O=top-right  I=bot-left  N=bot-right
const MAP = [
  [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
  [1,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,1],
  [1,0,1,1,0,0,0,1,1,0,0,1,1,0,0,0,1,1,0,1],
  [1,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,1],
  [1,1,0,1,0,1,0,0,1,1,1,1,0,0,1,0,1,0,1,1],
  [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
  [1,0,1,0,1,0,1,0,0,1,1,0,0,1,0,1,0,1,0,1],
  [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
  [1,0,1,0,1,0,1,0,0,1,1,0,0,1,0,1,0,1,0,1],
  [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
  [1,1,0,1,0,1,0,0,1,1,1,1,0,0,1,0,1,0,1,1],
  [1,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,1],
  [1,0,1,1,0,0,0,1,1,0,0,1,1,0,0,0,1,1,0,1],
  [1,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,1],
  [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
];

// Letters to collect, in order C → O → I → N
const LETTER_DATA = [
  { ch: 'C', col: 3,  row: 1  },
  { ch: 'O', col: 16, row: 1  },
  { ch: 'I', col: 3,  row: 13 },
  { ch: 'N', col: 16, row: 13 },
];

const PLAYER_START = { col: 10, row: 7 };

const ENEMY_STARTS = [
  { col: 7,  row: 5  },
  { col: 13, row: 5  },
  { col: 7,  row: 9  },
  { col: 13, row: 9  },
];

// ── HELPER ──────────────────────────────────────────
function walkable(col, row) {
  if (row < 0 || row >= ROWS || col < 0 || col >= COLS) return false;
  return MAP[row][col] === 0;
}

// =====================================================
//  MENU SCENE
// =====================================================
class MenuScene extends Phaser.Scene {
  constructor() { super('Menu'); }

  create() {
    // Background gradient
    const bg = this.add.graphics();
    bg.fillGradientStyle(0x1a3d0a, 0x1a3d0a, 0x0d1f05, 0x0d1f05, 1);
    bg.fillRect(0, 0, W, H);

    // Draw decorative trees in background
    for (let i = 0; i < 18; i++) {
      const x = 30 + (i % 6) * 110 + (Math.floor(i / 6) * 20);
      const y = 50 + Math.floor(i / 6) * 150;
      this._drawTreeDecor(bg, x, y, 0.4 + Math.random() * 0.3);
    }

    // Title
    this.add.text(W / 2, 80, 'FIND THE WORD', {
      fontSize: '46px',
      fontFamily: 'Georgia, serif',
      color: '#ffd700',
      stroke: '#4a2800',
      strokeThickness: 7,
    }).setOrigin(0.5);

    // Animated COIN letters
    const letterColors = ['#ff6b6b', '#ffd700', '#7bed9f', '#74b9ff'];
    const letterY = 160;
    ['C','O','I','N'].forEach((ch, i) => {
      const t = this.add.text(W / 2 - 90 + i * 62, letterY, ch, {
        fontSize: '60px',
        fontFamily: 'Georgia, serif',
        fontStyle: 'bold',
        color: letterColors[i],
        stroke: '#000000',
        strokeThickness: 5,
      }).setOrigin(0.5);

      this.tweens.add({
        targets: t,
        y: letterY - 18,
        duration: 550 + i * 80,
        yoyo: true,
        repeat: -1,
        ease: 'Sine.easeInOut',
        delay: i * 140,
      });
    });

    // Instructions box
    const boxX = W / 2 - 220, boxY = 248;
    const instrBg = this.add.graphics();
    instrBg.fillStyle(0x000000, 0.45);
    instrBg.fillRoundedRect(boxX, boxY, 440, 120, 12);

    const instrLines = [
      { txt: 'Collect letters  C → O → I → N  in order!', size: '17px', color: '#ffffff', dy: 20 },
      { txt: '⬆ ⬇ ⬅ ➡  or  W A S D  to move',            size: '15px', color: '#aaffaa', dy: 50 },
      { txt: 'Avoid the red enemies — you have 3 lives',  size: '15px', color: '#ffaaaa', dy: 78 },
    ];
    instrLines.forEach(l => {
      this.add.text(W / 2, boxY + l.dy, l.txt, {
        fontSize: l.size, color: l.color, fontFamily: 'Arial',
      }).setOrigin(0.5, 0);
    });

    // Start button
    const btn = this.add.text(W / 2, 410, '▶   START  GAME', {
      fontSize: '28px',
      fontFamily: 'Arial',
      fontStyle: 'bold',
      color: '#1a0000',
      backgroundColor: '#ffd700',
      padding: { x: 28, y: 12 },
    }).setOrigin(0.5).setInteractive({ useHandCursor: true });

    btn.on('pointerover',  () => btn.setStyle({ color: '#ffffff', backgroundColor: '#ff8c00' }));
    btn.on('pointerout',   () => btn.setStyle({ color: '#1a0000', backgroundColor: '#ffd700' }));
    btn.on('pointerdown',  () => this.scene.start('Game'));

    this.input.keyboard.once('keydown-SPACE', () => this.scene.start('Game'));
    this.input.keyboard.once('keydown-ENTER', () => this.scene.start('Game'));

    // Version credit
    this.add.text(W - 8, H - 8, 'Assets: Schwarnhild • Phaser 3', {
      fontSize: '10px', color: '#557744', fontFamily: 'Arial',
    }).setOrigin(1, 1);
  }

  _drawTreeDecor(gfx, x, y, alpha) {
    gfx.fillStyle(0x3d2b1f, alpha * 0.6);
    gfx.fillRect(x - 3, y + 8, 6, 14);
    gfx.fillStyle(0x2d7a1f, alpha * 0.5);
    gfx.fillCircle(x, y, 14);
    gfx.fillStyle(0x3da028, alpha * 0.4);
    gfx.fillCircle(x - 5, y - 3, 9);
    gfx.fillCircle(x + 5, y - 3, 9);
  }
}

// =====================================================
//  GAME SCENE
// =====================================================
class GameScene extends Phaser.Scene {
  constructor() { super('Game'); }

  // ── PRELOAD ────────────────────────────────────────
  preload() {
    // Player spritesheets  (96×96 per frame, 4 cols × 4 rows)
    this.load.spritesheet('player_walk', ASSETS + 'player/player-walk-96x96.png', {
      frameWidth: 96, frameHeight: 96,
    });
    this.load.spritesheet('player_idle', ASSETS + 'player/player-idle-96x96.png', {
      frameWidth: 96, frameHeight: 96,
    });

    // Loading bar
    const bar = this.add.graphics();
    this.load.on('progress', v => {
      bar.clear();
      bar.fillStyle(0x333333);
      bar.fillRect(W / 2 - 150, H / 2 - 10, 300, 20);
      bar.fillStyle(0xffd700);
      bar.fillRect(W / 2 - 150, H / 2 - 10, 300 * v, 20);
    });
    this.load.on('complete', () => bar.destroy());
  }

  // ── CREATE ─────────────────────────────────────────
  create() {
    this.lives       = 3;
    this.nextLetter  = 0;       // index 0-3 → C,O,I,N
    this.invincible  = false;
    this.gameEnded   = false;
    this._lastNext   = -1;      // for HUD pulse tracking

    this._buildWorld();
    this._createLetters();
    this._createPlayer();
    this._createEnemies();
    this._createHUD();

    this.cursors = this.input.keyboard.createCursorKeys();
    this.wasd    = this.input.keyboard.addKeys({
      up:    Phaser.Input.Keyboard.KeyCodes.W,
      down:  Phaser.Input.Keyboard.KeyCodes.S,
      left:  Phaser.Input.Keyboard.KeyCodes.A,
      right: Phaser.Input.Keyboard.KeyCodes.D,
    });
  }

  // ── BUILD WORLD ────────────────────────────────────
  _buildWorld() {
    const gfx = this.add.graphics().setDepth(0);

    for (let r = 0; r < ROWS; r++) {
      for (let c = 0; c < COLS; c++) {
        const x = c * TILE;
        const y = r * TILE;

        if (MAP[r][c] === 0) {
          // Grass — checkerboard shading + tiny detail dots
          const base = ((r + c) % 2 === 0) ? 0x4a7c23 : 0x518a26;
          gfx.fillStyle(base);
          gfx.fillRect(x, y, TILE, TILE);
          // Subtle blade highlights
          gfx.fillStyle(0x6aac3b, 0.25);
          gfx.fillRect(x + 3, y + 3, 4, 2);
          gfx.fillRect(x + 19, y + 15, 3, 2);
          gfx.fillRect(x + 11, y + 25, 2, 2);
        } else {
          // Tree tile — dark soil + foliage
          gfx.fillStyle(0x2f1e10);
          gfx.fillRect(x, y, TILE, TILE);
          // Trunk
          gfx.fillStyle(0x6b3a2a);
          gfx.fillRect(x + 13, y + 18, 6, 14);
          // Outer foliage rings
          gfx.fillStyle(0x1e5c10);
          gfx.fillCircle(x + 16, y + 14, 13);
          // Mid foliage
          gfx.fillStyle(0x2d7a1f);
          gfx.fillCircle(x + 12, y + 12, 8);
          gfx.fillCircle(x + 20, y + 12, 8);
          // Bright highlight
          gfx.fillStyle(0x4dac30);
          gfx.fillCircle(x + 16, y + 8, 6);
          gfx.fillStyle(0x6dc845, 0.5);
          gfx.fillCircle(x + 13, y + 7, 3);
        }
      }
    }
  }

  // ── LETTERS ────────────────────────────────────────
  _createLetters() {
    this.letterObjects = LETTER_DATA.map((ld, i) => {
      const x = ld.col * TILE + TILE / 2;
      const y = ld.row * TILE + TILE / 2;

      // Glow ring
      const glow = this.add.graphics().setDepth(2);
      glow.fillStyle(0xffd700, 0.25);
      glow.fillCircle(0, 0, 19);
      glow.lineStyle(2.5, 0xffd700, 0.9);
      glow.strokeCircle(0, 0, 19);
      glow.x = x; glow.y = y;

      // Letter
      const txt = this.add.text(x, y, ld.ch, {
        fontSize: '24px',
        fontFamily: 'Georgia, serif',
        fontStyle: 'bold',
        color: '#ffd700',
        stroke: '#7a4e00',
        strokeThickness: 3,
      }).setOrigin(0.5).setDepth(3);

      // Float tween
      this.tweens.add({
        targets: [txt, glow],
        y: '-=7',
        duration: 820 + i * 90,
        yoyo: true,
        repeat: -1,
        ease: 'Sine.easeInOut',
      });

      return { glow, txt, col: ld.col, row: ld.row, ch: ld.ch, alive: true };
    });
  }

  // ── PLAYER ─────────────────────────────────────────
  _createPlayer() {
    this.px = PLAYER_START.col * TILE + TILE / 2;
    this.py = PLAYER_START.row * TILE + TILE / 2;

    this.player = this.add.sprite(this.px, this.py, 'player_walk', 0)
      .setDepth(10)
      .setScale(TILE * 1.3 / 96);   // display ~41 px tall

    // Animations  (walk sheet: row0=down, row1=left, row2=right, row3=up)
    const makeDef = (key, sheet, start, end, rate, rpt) => {
      if (!this.anims.exists(key)) {
        this.anims.create({
          key,
          frames: this.anims.generateFrameNumbers(sheet, { start, end }),
          frameRate: rate,
          repeat: rpt,
        });
      }
    };
    makeDef('walk_down',  'player_walk',  0,  3, 8, -1);
    makeDef('walk_left',  'player_walk',  4,  7, 8, -1);
    makeDef('walk_right', 'player_walk',  8, 11, 8, -1);
    makeDef('walk_up',    'player_walk', 12, 15, 8, -1);
    makeDef('idle_down',  'player_idle',  0,  3, 4, -1);

    this.player.play('idle_down');
    this.pSpeed = 120;
  }

  // ── ENEMIES ────────────────────────────────────────
  _createEnemies() {
    this.enemies = ENEMY_STARTS.map((ed, i) => {
      const x = ed.col * TILE + TILE / 2;
      const y = ed.row * TILE + TILE / 2;
      const gfx = this.add.graphics().setDepth(9);
      return {
        gfx, x, y,
        speed: 52 + i * 6,
        dir: { x: 0, y: 1 },
        moveTimer: Math.random() * 1.2,
        idx: i,
      };
    });
  }

  _drawEnemy(gfx, offY, idx) {
    gfx.clear();
    // Shadow
    gfx.fillStyle(0x000000, 0.28);
    gfx.fillEllipse(0, 13 + offY, 22, 7);
    // Body
    const bodyColor = [0xcc2222, 0xcc4422, 0x991111, 0xaa3322][idx % 4];
    gfx.fillStyle(bodyColor);
    gfx.fillCircle(0, offY, 12);
    // Ear bumps
    gfx.fillStyle(Phaser.Display.Color.ValueToColor(bodyColor).lighten(15).color);
    gfx.fillCircle(-6, offY - 5, 7);
    gfx.fillCircle(6, offY - 5, 7);
    // White eyes
    gfx.fillStyle(0xffffff);
    gfx.fillCircle(-5, offY - 1, 4);
    gfx.fillCircle(5, offY - 1, 4);
    // Pupils
    gfx.fillStyle(0x110000);
    gfx.fillCircle(-5, offY - 1, 2);
    gfx.fillCircle(5, offY - 1, 2);
    // Gleam
    gfx.fillStyle(0xffffff, 0.7);
    gfx.fillCircle(-4, offY - 2, 1);
    gfx.fillCircle(6, offY - 2, 1);
  }

  // ── HUD ────────────────────────────────────────────
  _createHUD() {
    // Top bar background
    this.hudGfx = this.add.graphics().setDepth(50);
    this.hudGfx.fillStyle(0x000000, 0.55);
    this.hudGfx.fillRect(0, 0, W, 38);

    // Heart icons
    this.hearts = Array.from({ length: 3 }, (_, i) =>
      this.add.text(10 + i * 28, 7, '❤', {
        fontSize: '20px', color: '#ff4444',
      }).setDepth(51)
    );

    // Divider
    this.hudGfx.lineStyle(1, 0x556644, 0.6);
    this.hudGfx.lineBetween(W / 2 - 120, 6, W / 2 - 120, 32);
    this.hudGfx.lineBetween(W / 2 + 120, 6, W / 2 + 120, 32);

    // Word progress letters
    this.wordLetters = LETTER_DATA.map((ld, i) =>
      this.add.text(W / 2 - 55 + i * 38, 7, ld.ch, {
        fontSize: '22px',
        fontFamily: 'Georgia, serif',
        fontStyle: 'bold',
        color: '#444444',
        stroke: '#111111',
        strokeThickness: 2,
      }).setDepth(51)
    );

    // Hint label right side
    this.hintText = this.add.text(W - 10, 10, 'Next: C', {
      fontSize: '14px', color: '#aaaaaa', fontFamily: 'Arial',
    }).setOrigin(1, 0).setDepth(51);

    // Center message
    this.msgText = this.add.text(W / 2, H / 2, '', {
      fontSize: '40px',
      fontFamily: 'Georgia, serif',
      fontStyle: 'bold',
      color: '#ffd700',
      stroke: '#000000',
      strokeThickness: 5,
    }).setOrigin(0.5).setDepth(80).setAlpha(0);
  }

  _refreshHUD() {
    // Hearts
    this.hearts.forEach((h, i) => {
      h.setText(i < this.lives ? '❤' : '🖤');
      h.setColor(i < this.lives ? '#ff4444' : '#333333');
    });

    // Word progress
    this.wordLetters.forEach((t, i) => {
      if (i < this.nextLetter) {
        t.setStyle({ color: '#ffd700', stroke: '#7a4e00', strokeThickness: 2 });
        t.setScale(1);
      } else if (i === this.nextLetter) {
        t.setStyle({ color: '#ffffff', stroke: '#333333', strokeThickness: 2 });
        // Start pulse on new target
        if (this._lastNext !== this.nextLetter) {
          this.tweens.killTweensOf(t);
          this.tweens.add({
            targets: t,
            scaleX: 1.35, scaleY: 1.35,
            duration: 380,
            yoyo: true,
            repeat: -1,
            ease: 'Sine.easeInOut',
          });
        }
      } else {
        t.setStyle({ color: '#444444', stroke: '#111111', strokeThickness: 2 });
        t.setScale(1);
      }
    });

    // Hint
    if (this.nextLetter < LETTER_DATA.length) {
      this.hintText.setText(`Next: ${LETTER_DATA[this.nextLetter].ch}`);
    } else {
      this.hintText.setText('');
    }

    this._lastNext = this.nextLetter;
  }

  // ── UPDATE ─────────────────────────────────────────
  update(time, delta) {
    if (this.gameEnded) return;
    const dt = delta / 1000;

    this._movePlayer(dt);
    this._moveEnemies(dt, time);
    this._pickupCheck();
    this._enemyHitCheck();
    this._refreshHUD();
  }

  // ── PLAYER MOVEMENT ────────────────────────────────
  _movePlayer(dt) {
    const c = this.cursors, w = this.wasd;
    let vx = 0, vy = 0;

    if (c.left.isDown  || w.left.isDown)  vx = -1;
    if (c.right.isDown || w.right.isDown) vx =  1;
    if (c.up.isDown    || w.up.isDown)    vy = -1;
    if (c.down.isDown  || w.down.isDown)  vy =  1;

    // Normalise diagonal
    if (vx !== 0 && vy !== 0) { vx *= 0.707; vy *= 0.707; }

    const spd = this.pSpeed * dt;
    const R   = 11;

    // Horizontal
    const nx = this.px + vx * spd;
    if (vx !== 0) {
      const r1 = Math.floor((this.py - R) / TILE);
      const r2 = Math.floor((this.py + R) / TILE);
      const lc = Math.floor((nx - R) / TILE);
      const rc = Math.floor((nx + R) / TILE);
      if (walkable(lc, r1) && walkable(lc, r2) && walkable(rc, r1) && walkable(rc, r2)) {
        this.px = nx;
      }
    }

    // Vertical
    const ny = this.py + vy * spd;
    if (vy !== 0) {
      const c1 = Math.floor((this.px - R) / TILE);
      const c2 = Math.floor((this.px + R) / TILE);
      const tr = Math.floor((ny - R) / TILE);
      const br = Math.floor((ny + R) / TILE);
      if (walkable(c1, tr) && walkable(c2, tr) && walkable(c1, br) && walkable(c2, br)) {
        this.py = ny;
      }
    }

    this.player.x = this.px;
    this.player.y = this.py;

    // Animation
    if (vx < 0)      this.player.play('walk_left',  true);
    else if (vx > 0) this.player.play('walk_right', true);
    else if (vy < 0) this.player.play('walk_up',    true);
    else if (vy > 0) this.player.play('walk_down',  true);
    else             this.player.play('idle_down',  true);
  }

  // ── ENEMY MOVEMENT ─────────────────────────────────
  _moveEnemies(dt, time) {
    this.enemies.forEach(e => {
      e.moveTimer -= dt;

      const dx   = this.px - e.x;
      const dy   = this.py - e.y;
      const dist = Math.hypot(dx, dy);

      if (dist < 190) {
        // Chase
        e.dir.x = dx / dist;
        e.dir.y = dy / dist;
        e.moveTimer = Math.max(e.moveTimer, 0.05);
      } else if (e.moveTimer <= 0) {
        // Wander
        const ang = Math.random() * Math.PI * 2;
        e.dir.x = Math.cos(ang);
        e.dir.y = Math.sin(ang);
        e.moveTimer = 0.7 + Math.random() * 1.3;
      }

      const spd = e.speed * dt;
      const R   = 9;

      // Horizontal
      const nx = e.x + e.dir.x * spd;
      if (walkable(Math.floor((nx - R) / TILE), Math.floor((e.y - R) / TILE)) &&
          walkable(Math.floor((nx - R) / TILE), Math.floor((e.y + R) / TILE)) &&
          walkable(Math.floor((nx + R) / TILE), Math.floor((e.y - R) / TILE)) &&
          walkable(Math.floor((nx + R) / TILE), Math.floor((e.y + R) / TILE))) {
        e.x = nx;
      } else {
        e.dir.x *= -1;
      }

      // Vertical
      const ny = e.y + e.dir.y * spd;
      if (walkable(Math.floor((e.x - R) / TILE), Math.floor((ny - R) / TILE)) &&
          walkable(Math.floor((e.x - R) / TILE), Math.floor((ny + R) / TILE)) &&
          walkable(Math.floor((e.x + R) / TILE), Math.floor((ny - R) / TILE)) &&
          walkable(Math.floor((e.x + R) / TILE), Math.floor((ny + R) / TILE))) {
        e.y = ny;
      } else {
        e.dir.y *= -1;
      }

      e.gfx.x = e.x;
      e.gfx.y = e.y;

      // Animated bob
      const bob = Math.sin(time * 0.004 + e.idx * 1.2) * 2.5;
      this._drawEnemy(e.gfx, bob, e.idx);
    });
  }

  // ── LETTER PICKUP ──────────────────────────────────
  _pickupCheck() {
    if (this.nextLetter >= LETTER_DATA.length) return;

    const target = this.letterObjects[this.nextLetter];
    if (!target.alive) return;

    const dist = Math.hypot(
      this.px - (target.col * TILE + TILE / 2),
      this.py - (target.row * TILE + TILE / 2)
    );

    if (dist < 22) {
      target.alive = false;
      target.txt.destroy();
      target.glow.destroy();

      const ch = LETTER_DATA[this.nextLetter].ch;
      this.nextLetter++;

      // Particle burst (simple)
      this._burstParticles(target.col * TILE + TILE / 2, target.row * TILE + TILE / 2);

      if (this.nextLetter >= LETTER_DATA.length) {
        this._winGame();
      } else {
        this._flashMsg(`'${ch}' collected!`, '#ffd700');
      }
    }
  }

  // ── ENEMY HIT ──────────────────────────────────────
  _enemyHitCheck() {
    if (this.invincible) return;

    for (const e of this.enemies) {
      if (Math.hypot(this.px - e.x, this.py - e.y) < 22) {
        this.lives--;
        this.invincible = true;

        // Flash white
        this.player.setTint(0xff8888);
        this.tweens.add({
          targets: this.player,
          alpha: 0.3,
          duration: 90,
          yoyo: true,
          repeat: 6,
          onComplete: () => {
            this.player.clearTint().setAlpha(1);
            this.invincible = false;
          },
        });

        if (this.lives <= 0) {
          this._gameOver();
        } else {
          this._flashMsg('Ouch! –1 life', '#ff5555');
        }
        break;
      }
    }
  }

  // ── PARTICLE BURST ─────────────────────────────────
  _burstParticles(x, y) {
    for (let i = 0; i < 10; i++) {
      const ang = (i / 10) * Math.PI * 2;
      const dot = this.add.graphics().setDepth(20);
      dot.fillStyle(0xffd700);
      dot.fillCircle(0, 0, 3);
      dot.x = x; dot.y = y;
      this.tweens.add({
        targets: dot,
        x: x + Math.cos(ang) * 35,
        y: y + Math.sin(ang) * 35,
        alpha: 0,
        scaleX: 0.2, scaleY: 0.2,
        duration: 450,
        ease: 'Power2.easeOut',
        onComplete: () => dot.destroy(),
      });
    }
  }

  // ── FLASH MESSAGE ──────────────────────────────────
  _flashMsg(text, color) {
    this.tweens.killTweensOf(this.msgText);
    this.msgText.setText(text).setColor(color).setAlpha(1);
    this.tweens.add({
      targets: this.msgText,
      alpha: 0,
      delay: 700,
      duration: 900,
    });
  }

  // ── WIN ────────────────────────────────────────────
  _winGame() {
    this.gameEnded = true;

    // Stop enemies
    this.enemies.forEach(e => e.speed = 0);

    const overlay = this.add.graphics().setDepth(90);
    overlay.fillStyle(0x000000, 0.6);
    overlay.fillRect(0, 0, W, H);

    // Trophy stars
    for (let i = 0; i < 20; i++) {
      const star = this.add.text(
        20 + Math.random() * (W - 40),
        20 + Math.random() * (H - 40),
        '⭐', { fontSize: `${12 + Math.random() * 16}px` }
      ).setDepth(91).setAlpha(0);
      this.tweens.add({
        targets: star, alpha: 1, delay: i * 80, duration: 300,
        yoyo: true, repeat: -1,
      });
    }

    this.add.text(W / 2, H / 2 - 95, '🎉  YOU FOUND IT!', {
      fontSize: '40px', fontFamily: 'Georgia', color: '#ffd700',
      stroke: '#000000', strokeThickness: 5,
    }).setOrigin(0.5).setDepth(92);

    // Animate each letter flying in
    const word = ['C','O','I','N'];
    const colors = ['#ff6b6b','#ffd700','#7bed9f','#74b9ff'];
    word.forEach((ch, i) => {
      const t = this.add.text(W / 2 - 75 + i * 52, H / 2 - 20, ch, {
        fontSize: '72px', fontFamily: 'Georgia', fontStyle: 'bold',
        color: colors[i], stroke: '#000', strokeThickness: 5,
      }).setOrigin(0.5).setDepth(92).setAlpha(0).setScale(0.2);

      this.tweens.add({
        targets: t, alpha: 1, scale: 1,
        delay: 200 + i * 180,
        duration: 400,
        ease: 'Back.easeOut',
      });
    });

    const btn = this.add.text(W / 2, H / 2 + 100, 'Play Again', {
      fontSize: '26px', color: '#000000', backgroundColor: '#ffd700',
      padding: { x: 22, y: 10 },
    }).setOrigin(0.5).setDepth(92).setInteractive({ useHandCursor: true });

    btn.on('pointerover', () => btn.setStyle({ color: '#fff', backgroundColor: '#ff8c00' }));
    btn.on('pointerout',  () => btn.setStyle({ color: '#000', backgroundColor: '#ffd700' }));
    btn.on('pointerdown', () => this.scene.restart());
    this.input.keyboard.once('keydown-SPACE', () => this.scene.restart());
  }

  // ── GAME OVER ──────────────────────────────────────
  _gameOver() {
    this.gameEnded = true;
    this.enemies.forEach(e => e.speed = 0);

    const overlay = this.add.graphics().setDepth(90);
    overlay.fillStyle(0x330000, 0.75);
    overlay.fillRect(0, 0, W, H);

    this.add.text(W / 2, H / 2 - 70, 'GAME OVER', {
      fontSize: '54px', fontFamily: 'Georgia', color: '#ff4444',
      stroke: '#000000', strokeThickness: 7,
    }).setOrigin(0.5).setDepth(92);

    const collected = LETTER_DATA.slice(0, this.nextLetter).map(l => l.ch).join('') || '—';
    this.add.text(W / 2, H / 2 + 5, `Letters collected: ${collected}`, {
      fontSize: '22px', color: '#ffaaaa', fontFamily: 'Arial',
    }).setOrigin(0.5).setDepth(92);

    const btn = this.add.text(W / 2, H / 2 + 75, 'Try Again', {
      fontSize: '26px', color: '#ffffff', backgroundColor: '#880000',
      padding: { x: 22, y: 10 },
    }).setOrigin(0.5).setDepth(92).setInteractive({ useHandCursor: true });

    btn.on('pointerover', () => btn.setStyle({ backgroundColor: '#cc0000' }));
    btn.on('pointerout',  () => btn.setStyle({ backgroundColor: '#880000' }));
    btn.on('pointerdown', () => this.scene.restart());
    this.input.keyboard.once('keydown-SPACE', () => this.scene.restart());
  }
}

// =====================================================
//  PHASER GAME CONFIG
// =====================================================
const config = {
  type: Phaser.AUTO,
  width:  W,
  height: H,
  backgroundColor: '#1a3d0a',
  parent: 'game-container',
  scene: [MenuScene, GameScene],
  audio: { noAudio: true },
};

new Phaser.Game(config);
