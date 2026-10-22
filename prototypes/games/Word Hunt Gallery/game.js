'use strict';

// ═══════════════════════════════════════════════════════════════
//  WORD HUNT GALLERY
//  A Carnival Shooting Gallery for Speech & Language Practice
//  Uses Kenney Shooting Gallery assets
//
//  5 rounds of language therapy:
//    1. Animals        4. Body Parts
//    2. Starts with S  5. Things in a House
//    3. Foods
//
//  Shoot the CORRECT words — avoid the WRONG ones!
//  12 bullets • 40-second timer • +100 per correct hit
// ═══════════════════════════════════════════════════════════════

const GW   = 900;
const GH   = 600;
const BASE = '../New folder/kenney_shooting-gallery/PNG';

const MAX_AMMO       = 12;
const ROUND_DURATION = 40; // seconds

// Rail y-centres for the 3 duck rows
const RAIL_Y = [205, 310, 415];

// ── Language-therapy rounds ──────────────────────────────────────
const ROUNDS = [
  {
    headline: '🐾  SHOOT THE ANIMALS!',
    sub:      'Click on anything that is an ANIMAL',
    targets:     ['DOG','CAT','BIRD','FISH','BEAR','COW'],
    distractors: ['APPLE','CHAIR','SHOE','BALL','TREE','BOOK'],
  },
  {
    headline: '🔤  SHOOT WORDS STARTING WITH "S"!',
    sub:      'Click words that start with the letter  S',
    targets:     ['SUN','SOCK','SEAL','STAR','SOUP','SAND'],
    distractors: ['DOG','BALL','HAT','FROG','CAKE','BOAT'],
  },
  {
    headline: '🍎  SHOOT FOODS!',
    sub:      'Click things you EAT or DRINK',
    targets:     ['CAKE','MILK','APPLE','BREAD','SOUP','FISH'],
    distractors: ['CHAIR','BOAT','SHOE','TREE','BALL','ROCK'],
  },
  {
    headline: '🖐  SHOOT BODY PARTS!',
    sub:      'Click parts of YOUR BODY',
    targets:     ['HAND','FOOT','EAR','NOSE','ARM','KNEE'],
    distractors: ['CAT','BALL','CAKE','TREE','CAR','BOAT'],
  },
  {
    headline: '🏠  SHOOT THINGS IN A HOUSE!',
    sub:      'Click things found INSIDE a HOUSE',
    targets:     ['CHAIR','TABLE','LAMP','BED','DOOR','SINK'],
    distractors: ['FISH','BIRD','APPLE','CLOUD','ROCK','FROG'],
  },
];

// ───────────────────────────────────────────────────────────────
//  BOOT SCENE — asset preload
// ───────────────────────────────────────────────────────────────
class BootScene extends Phaser.Scene {
  constructor() { super('BootScene'); }

  preload() {
    // Progress bar
    this.add.rectangle(GW / 2, GH / 2, 504, 34, 0x333333);
    const bar = this.add.rectangle(GW / 2 - 250, GH / 2, 0, 30, 0xFF7700).setOrigin(0, 0.5);
    this.add.text(GW / 2, GH / 2 - 50, 'Loading…', {
      fontFamily: 'Arial', fontSize: '20px', color: '#ffffff'
    }).setOrigin(0.5);
    this.load.on('progress', v => bar.setDisplaySize(500 * v, 30));

    const O = `${BASE}/Objects`;
    const H = `${BASE}/HUD`;
    const S = `${BASE}/Stall`;

    // Ducks
    this.load.image('duck_yellow',  `${O}/duck_yellow.png`);
    this.load.image('duck_brown',   `${O}/duck_brown.png`);
    this.load.image('duck_white',   `${O}/duck_white.png`);

    // Stick & shot effects
    this.load.image('stick_wood',   `${O}/stick_woodFixed.png`);
    this.load.image('shot_yellow',  `${O}/shot_yellow_large.png`);
    this.load.image('shot_blue',    `${O}/shot_blue_large.png`);

    // HUD
    this.load.image('crosshair',    `${H}/crosshair_red_large.png`);
    this.load.image('bullet_gold',  `${H}/icon_bullet_gold_long.png`);
    this.load.image('bullet_empty', `${H}/icon_bullet_empty_long.png`);
    this.load.image('txt_ready',    `${H}/text_ready.png`);
    this.load.image('txt_go',       `${H}/text_go.png`);
    this.load.image('txt_timeup',   `${H}/text_timeup.png`);
    this.load.image('txt_gameover', `${H}/text_gameover.png`);

    // Stall
    this.load.image('bg_blue',      `${S}/bg_blue.png`);
    this.load.image('curtain_top',  `${S}/curtain_top.png`);
    this.load.image('grass1',       `${S}/grass1.png`);
    this.load.image('tree_pine',    `${S}/tree_pine.png`);
    this.load.image('tree_oak',     `${S}/tree_oak.png`);
    this.load.image('cloud1',       `${S}/cloud1.png`);
  }

  create() { this.scene.start('MenuScene'); }
}

// ───────────────────────────────────────────────────────────────
//  MENU SCENE
// ───────────────────────────────────────────────────────────────
class MenuScene extends Phaser.Scene {
  constructor() { super('MenuScene'); }

  create() {
    this._bg();

    // ── Title ──────────────────────────────────────────────
    this.add.text(GW / 2, 108, 'WORD HUNT', {
      fontFamily: 'Impact, Arial Black, sans-serif',
      fontSize:   '78px',
      color:      '#FF7700',
      stroke:     '#5D1F00',
      strokeThickness: 8,
      shadow: { offsetX: 4, offsetY: 4, color: '#00000080', blur: 6, fill: true }
    }).setOrigin(0.5);

    this.add.text(GW / 2, 196, 'GALLERY', {
      fontFamily: 'Impact, Arial Black, sans-serif',
      fontSize:   '52px',
      color:      '#FFE066',
      stroke:     '#8B4513',
      strokeThickness: 6
    }).setOrigin(0.5);

    this.add.text(GW / 2, 258, 'A Language Practice Shooting Gallery', {
      fontFamily: 'Arial', fontSize: '16px', color: '#DDDDDD'
    }).setOrigin(0.5);

    // ── Bouncing ducks ─────────────────────────────────────
    ['duck_yellow', 'duck_brown', 'duck_white'].forEach((k, i) => {
      const img = this.add.image(240 + i * 210, 342, k).setScale(0.9);
      this.tweens.add({
        targets: img, y: 328, yoyo: true, repeat: -1,
        duration: 600 + i * 140, ease: 'Sine.InOut'
      });
    });
    this.add.image(GW / 2, 342, 'crosshair').setScale(1.1).setAlpha(0.45);

    // ── Instructions ───────────────────────────────────────
    this.add.text(GW / 2, 435, [
      '5 rounds  •  12 bullets each  •  40-second timer',
      '',
      '1. Animals    2. Starts with "S"    3. Foods',
      '4. Body Parts    5. Things in a House',
      '',
      'Move your crosshair and CLICK to shoot the correct words!'
    ].join('\n'), {
      fontFamily: 'Arial', fontSize: '14px',
      color: '#E0E0E0', lineSpacing: 4, align: 'center'
    }).setOrigin(0.5);

    // ── Play button ────────────────────────────────────────
    const btn = this.add.rectangle(GW / 2, 549, 295, 66, 0xFF7700)
      .setInteractive({ useHandCursor: true });
    this.add.text(GW / 2, 549, '▶  START SHOOTING!', {
      fontFamily: 'Arial Black', fontSize: '22px', color: '#ffffff'
    }).setOrigin(0.5);

    btn.on('pointerover', () => btn.setFillStyle(0xFF9933));
    btn.on('pointerout',  () => btn.setFillStyle(0xFF7700));
    btn.on('pointerdown', () => {
      this.scene.start('GameScene', { round: 0, score: 0 });
    });
  }

  _bg() {
    for (let x = 0; x < GW; x += 256)
      this.add.image(x, GH / 2, 'bg_blue').setOrigin(0, 0.5).setDisplaySize(256, GH);
    for (let x = 0; x < GW; x += 180)
      this.add.image(x, 0, 'curtain_top').setOrigin(0, 0).setDisplaySize(180, 58);
    for (let x = 0; x < GW; x += 200)
      this.add.image(x, GH, 'grass1').setOrigin(0, 1).setDisplaySize(200, 110);
    this.add.image(60,       GH - 106, 'tree_pine').setScale(0.28).setAlpha(0.7);
    this.add.image(GW - 60,  GH - 106, 'tree_oak').setScale(0.24).setAlpha(0.7);
    this.add.image(155, 148, 'cloud1').setScale(0.55).setAlpha(0.5);
    this.add.image(745, 132, 'cloud1').setScale(0.42).setAlpha(0.45);
  }
}

// ───────────────────────────────────────────────────────────────
//  GAME SCENE — handles all 5 rounds (one scene visit per round)
// ───────────────────────────────────────────────────────────────
class GameScene extends Phaser.Scene {
  constructor() { super('GameScene'); }

  init(data) {
    this.roundIdx    = data.round  ?? 0;
    this.totalScore  = data.score  ?? 0;
    this.roundScore  = 0;
    this.bullets     = MAX_AMMO;
    this.timeLeft    = ROUND_DURATION;
    this.phase       = 'READY';   // READY → PLAYING → ENDED
    this.canShoot    = false;
    this.ducks       = [];
    this.correctHits = 0;
    this.wrongHits   = 0;
    this.timerEvent  = null;
  }

  create() {
    this.roundData = ROUNDS[this.roundIdx];

    // Restore cursor on scene shutdown (safety)
    this.events.once('shutdown', () => {
      this.game.canvas.style.cursor = 'default';
    });

    // ── Static scene pieces ──────────────────────────────
    this._buildBackground();
    this._buildRails();

    // ── Duck targets ─────────────────────────────────────
    this._spawnInitialDucks();

    // ── HUD overlay ──────────────────────────────────────
    this._buildHUD();

    // ── Screen-flash overlay (non-interactive) ────────────
    this.flashOver = this.add.rectangle(GW / 2, GH / 2, GW, GH, 0xFFFFFF)
      .setDepth(75).setAlpha(0);

    // ── Custom crosshair ──────────────────────────────────
    this.crosshair = this.add.image(GW / 2, GH / 2, 'crosshair')
      .setScale(0.85).setDepth(100).setAlpha(0);

    this.input.on('pointermove', p => this.crosshair.setPosition(p.x, p.y));
    this.input.on('pointerdown', p => {
      if (this.canShoot) this._shoot(p.x, p.y);
    });
    this.game.canvas.style.cursor = 'none';

    // ── Begin: READY → GO ─────────────────────────────────
    this._showReadyGo();
  }

  // ── Background + decoration ────────────────────────────────
  _buildBackground() {
    for (let x = 0; x < GW; x += 256)
      this.add.image(x, GH / 2 + 20, 'bg_blue')
        .setOrigin(0, 0.5).setDisplaySize(256, GH - 80).setDepth(0);
    for (let x = 0; x < GW; x += 180)
      this.add.image(x, 0, 'curtain_top')
        .setOrigin(0, 0).setDisplaySize(180, 58).setDepth(2);
    for (let x = 0; x < GW; x += 200)
      this.add.image(x, GH, 'grass1')
        .setOrigin(0, 1).setDisplaySize(200, 110).setDepth(2);
    this.add.image(48,       GH - 108, 'tree_pine').setScale(0.26).setAlpha(0.65).setDepth(1);
    this.add.image(GW - 48,  GH - 108, 'tree_oak').setScale(0.22).setAlpha(0.65).setDepth(1);
  }

  // ── Rail shelves (3 horizontal tracks) ────────────────────
  _buildRails() {
    RAIL_Y.forEach(y => {
      this.add.rectangle(GW / 2, y + 52, GW, 8,  0x6B3A1F).setDepth(3);
      this.add.rectangle(GW / 2, y + 57, GW, 4,  0x3D1F08, 0.45).setDepth(3);
    });
  }

  // ── Spawn 12 ducks (6 targets + 6 distractors) ────────────
  _spawnInitialDucks() {
    const words = Phaser.Utils.Array.Shuffle([
      ...this.roundData.targets.map(w     => ({ word: w, isTarget: true  })),
      ...this.roundData.distractors.map(w => ({ word: w, isTarget: false })),
    ]);

    let idx = 0;
    RAIL_Y.forEach((railY, ri) => {
      for (let j = 0; j < 4 && idx < words.length; j++, idx++) {
        const x     = 100 + j * 200;
        const spd   = (1.4 + Math.random() * 1.6) * (ri % 2 === 0 ? 1 : -1);
        this._createDuck(words[idx].word, words[idx].isTarget, x, railY, spd);
      }
    });
  }

  // ── Build one duck container ───────────────────────────────
  _createDuck(word, isTarget, x, railY, speed) {
    const cont = this.add.container(x, railY).setDepth(10);

    // Wooden stick below
    const stick = this.add.image(0, 52, 'stick_wood').setDisplaySize(18, 92);

    // Duck image — targets: yellow/white, distractors: brown
    const key  = isTarget
      ? (Math.random() < 0.5 ? 'duck_yellow' : 'duck_white')
      : 'duck_brown';
    const img  = this.add.image(0, 0, key).setDisplaySize(72, 72);

    // Word label above the duck
    const lW   = Math.max(72, word.length * 11 + 14);
    const lBg  = this.add.rectangle(0, -50, lW, 26, 0xFFFFFF, 0.93)
      .setStrokeStyle(2, 0x2C2C2C);
    const lTxt = this.add.text(0, -50, word, {
      fontFamily: 'Arial Black', fontSize: '13px', color: '#111111'
    }).setOrigin(0.5);

    cont.add([stick, img, lBg, lTxt]);

    // Gentle bobbing animation on the duck sprite
    const bobTween = this.tweens.add({
      targets: img, y: -7,
      yoyo: true, repeat: -1,
      duration: 640 + Math.random() * 360,
      ease: 'Sine.InOut'
    });

    const duck = { cont, img, word, isTarget, speed, railY,
                   alive: true, shaking: false, bobTween };
    this.ducks.push(duck);
    return duck;
  }

  // ── HUD ────────────────────────────────────────────────────
  _buildHUD() {
    this.add.rectangle(GW / 2, 36, GW, 72, 0x111111, 0.88).setDepth(50);

    // Score (left)
    this.scoreTxt = this.add.text(16, 36, `SCORE  ${this.totalScore}`, {
      fontFamily: 'Arial Black', fontSize: '19px', color: '#FFD700'
    }).setOrigin(0, 0.5).setDepth(51);

    // Round counter (centre-top)
    this.add.text(GW / 2, 16, `Round  ${this.roundIdx + 1}  /  ${ROUNDS.length}`, {
      fontFamily: 'Arial', fontSize: '13px', color: '#AAAAAA'
    }).setOrigin(0.5).setDepth(51);

    // Category headline (centre)
    this.add.text(GW / 2, 44, this.roundData.headline, {
      fontFamily: 'Arial Black', fontSize: '15px', color: '#FFE066',
      stroke: '#000000', strokeThickness: 2
    }).setOrigin(0.5).setDepth(51);

    // Timer (right)
    this.timerTxt = this.add.text(GW - 14, 18, `${this.timeLeft}s`, {
      fontFamily: 'Arial Black', fontSize: '19px', color: '#FF8888'
    }).setOrigin(1, 0.5).setDepth(51);

    // Bullet icons (right, second row)
    this.bulletIcons = [];
    for (let i = 0; i < MAX_AMMO; i++) {
      const ic = this.add.image(GW - 14 - (MAX_AMMO - 1 - i) * 15, 54, 'bullet_gold')
        .setScale(0.55).setDepth(51);
      this.bulletIcons.push(ic);
    }

    // Sub-prompt strip just below HUD bar
    this.add.text(GW / 2, 82, this.roundData.sub, {
      fontFamily: 'Arial', fontSize: '13px', color: '#CCCCCC',
      backgroundColor: '#00000060', padding: { x: 10, y: 4 }
    }).setOrigin(0.5).setDepth(51);
  }

  // ── READY → GO sequence ─────────────────────────────────────
  _showReadyGo() {
    const rd = this.add.image(GW / 2, GH / 2, 'txt_ready')
      .setScale(1.6).setDepth(90);

    this.time.delayedCall(1100, () => {
      this.tweens.add({
        targets: rd, alpha: 0, duration: 280,
        onComplete: () => rd.destroy()
      });
      const go = this.add.image(GW / 2, GH / 2, 'txt_go')
        .setScale(1.6).setDepth(90);
      this.tweens.add({
        targets: go, scaleX: 2.5, scaleY: 2.5, alpha: 0,
        duration: 650, ease: 'Power2',
        onComplete: () => {
          go.destroy();
          this.phase    = 'PLAYING';
          this.canShoot = true;
          this.crosshair.setAlpha(1);
          this._startTimer();
        }
      });
    });
  }

  // ── Countdown timer ─────────────────────────────────────────
  _startTimer() {
    this.timerEvent = this.time.addEvent({
      delay:    1000,
      repeat:   this.timeLeft - 1,
      callback: () => {
        this.timeLeft = Math.max(0, this.timeLeft - 1);
        this.timerTxt.setText(`${this.timeLeft}s`);
        if (this.timeLeft <= 10) this.timerTxt.setColor('#FF2222');
        if (this.timeLeft === 0) this._onRoundEnd('time');
      }
    });
  }

  // ── Shoot ───────────────────────────────────────────────────
  _shoot(wx, wy) {
    if (this.bullets <= 0) return;

    this.bullets--;
    this._updateAmmoHUD();

    // Muzzle flash at cursor
    const flash = this.add.image(wx, wy, 'shot_blue').setScale(0.45).setDepth(60);
    this.tweens.add({
      targets: flash, alpha: 0, scaleX: 1.1, scaleY: 1.1,
      duration: 340, onComplete: () => flash.destroy()
    });

    // Check duck hit — rectangular zone around duck + label
    let hit = null;
    for (const d of this.ducks) {
      if (!d.alive || d.shaking) continue;
      const dx = Math.abs(d.cont.x - wx);
      const dy = Math.abs((d.cont.y - 25) - wy); // -25 centres between duck and label
      if (dx < 56 && dy < 62) { hit = d; break; }
    }

    if (hit) {
      hit.isTarget ? this._correctHit(hit, wx, wy) : this._wrongHit(hit, wx, wy);
    }

    if (this.bullets <= 0) {
      this.time.delayedCall(420, () => this._onRoundEnd('ammo'));
    }
  }

  _correctHit(d, wx, wy) {
    d.alive = false;
    d.bobTween.stop();
    this.correctHits++;

    this.roundScore += 100;
    this._refreshScore();

    // Shot hole in local coords
    const hole = this.add.image(wx - d.cont.x, wy - d.cont.y, 'shot_yellow').setScale(0.48);
    d.cont.add(hole);

    // Duck falls and fades
    this.tweens.add({
      targets:  d.cont,
      y:        d.cont.y + 145,
      rotation: (Math.random() < 0.5 ? 1 : -1) * 0.5,
      alpha:    0,
      duration: 560,
      ease:     'Power2',
      onComplete: () => {
        d.cont.destroy();
        this.ducks = this.ducks.filter(x => x !== d);
        if (this.phase === 'PLAYING') {
          this.time.delayedCall(3200, () => this._respawnDuck(d.railY));
        }
      }
    });

    this._flashScreen(0x00DD00);
    this._popup(wx, wy - 30, '+100', '#44FF44');
  }

  _wrongHit(d, wx, wy) {
    d.shaking = true;
    this.wrongHits++;

    const origX = d.cont.x;
    this.tweens.add({
      targets:  d.cont,
      x:        origX + 14,
      yoyo:     true,
      repeat:   3,
      duration: 55,
      onComplete: () => { d.cont.x = origX; d.shaking = false; }
    });

    this._flashScreen(0xFF2222);
    this._popup(wx, wy - 30, '✗  WRONG', '#FF5555');
  }

  // ── Respawn a duck on a given rail ──────────────────────────
  _respawnDuck(railY) {
    if (this.phase !== 'PLAYING') return;
    const pool = [
      ...this.roundData.targets.map(w     => ({ word: w, isTarget: true  })),
      ...this.roundData.distractors.map(w => ({ word: w, isTarget: false })),
    ];
    const pick = pool[Math.floor(Math.random() * pool.length)];
    const side = Math.random() < 0.5 ? -85 : GW + 85;
    const spd  = (1.4 + Math.random() * 1.6) * (side < 0 ? 1 : -1);
    this._createDuck(pick.word, pick.isTarget, side, railY, spd);
  }

  // ── HUD helpers ─────────────────────────────────────────────
  _updateAmmoHUD() {
    this.bulletIcons.forEach((ic, i) => {
      ic.setTexture(i < this.bullets ? 'bullet_gold' : 'bullet_empty');
    });
  }

  _refreshScore() {
    this.scoreTxt.setText(`SCORE  ${this.totalScore + this.roundScore}`);
  }

  _flashScreen(color) {
    this.flashOver.setFillStyle(color).setAlpha(0.24);
    this.tweens.killTweensOf(this.flashOver);
    this.tweens.add({ targets: this.flashOver, alpha: 0, duration: 290 });
  }

  _popup(x, y, msg, color) {
    const txt = this.add.text(x, y, msg, {
      fontFamily: 'Arial Black', fontSize: '20px',
      color, stroke: '#000000', strokeThickness: 3
    }).setOrigin(0.5).setDepth(70);
    this.tweens.add({
      targets: txt, y: y - 48, alpha: 0,
      duration: 700, ease: 'Power1',
      onComplete: () => txt.destroy()
    });
  }

  // ── Main update loop (duck movement) ───────────────────────
  update() {
    if (this.phase !== 'PLAYING') return;
    for (const d of this.ducks) {
      if (!d.alive || d.shaking) continue;
      d.cont.x += d.speed;
      if (d.cont.x >  GW + 90) d.speed = -Math.abs(d.speed);
      if (d.cont.x < -90     ) d.speed =  Math.abs(d.speed);
      d.img.setFlipX(d.speed < 0);
    }
  }

  // ── Round end ───────────────────────────────────────────────
  _onRoundEnd(reason) {
    if (this.phase === 'ENDED') return;
    this.phase    = 'ENDED';
    this.canShoot = false;
    this.timerEvent && this.timerEvent.remove(false);
    this.crosshair.setAlpha(0);
    this.game.canvas.style.cursor = 'default';

    if (reason === 'time') {
      const img = this.add.image(GW / 2, GH / 2, 'txt_timeup')
        .setScale(1.5).setDepth(90).setAlpha(0);
      this.tweens.add({ targets: img, alpha: 1, duration: 350 });
      this.time.delayedCall(1700, () => { img.destroy(); this._showResult(); });
    } else {
      this.time.delayedCall(300, () => this._showResult());
    }
  }

  _showResult() {
    const newTotal = this.totalScore + this.roundScore;
    const isLast   = (this.roundIdx + 1) >= ROUNDS.length;

    this.add.rectangle(GW / 2, GH / 2, GW * 0.62, 240, 0x111111, 0.94)
      .setDepth(91).setStrokeStyle(3, 0xFF7700);

    this.add.text(GW / 2, GH / 2 - 86, `⭐  Round ${this.roundIdx + 1} Complete!  ⭐`, {
      fontFamily: 'Arial Black', fontSize: '24px', color: '#FF7700'
    }).setOrigin(0.5).setDepth(92);

    this.add.text(GW / 2, GH / 2 - 42, `✅ Correct: ${this.correctHits}     ❌ Wrong: ${this.wrongHits}`, {
      fontFamily: 'Arial', fontSize: '16px', color: '#CCCCCC'
    }).setOrigin(0.5).setDepth(92);

    this.add.text(GW / 2, GH / 2 + 0, `Round Score:  +${this.roundScore}`, {
      fontFamily: 'Arial Black', fontSize: '21px', color: '#FFD700'
    }).setOrigin(0.5).setDepth(92);

    this.add.text(GW / 2, GH / 2 + 38, `Total Score:  ${newTotal}`, {
      fontFamily: 'Arial', fontSize: '16px', color: '#FFFFFF'
    }).setOrigin(0.5).setDepth(92);

    const lbl = isLast ? '🏆  FINAL SCORE' : `ROUND  ${this.roundIdx + 2}  ▶`;
    const btn = this.add.rectangle(GW / 2, GH / 2 + 88, 255, 52, 0xFF7700)
      .setInteractive({ useHandCursor: true }).setDepth(92);
    this.add.text(GW / 2, GH / 2 + 88, lbl, {
      fontFamily: 'Arial Black', fontSize: '18px', color: '#ffffff'
    }).setOrigin(0.5).setDepth(93);

    btn.on('pointerover', () => btn.setFillStyle(0xFF9933));
    btn.on('pointerout',  () => btn.setFillStyle(0xFF7700));
    btn.on('pointerdown', () => {
      if (isLast) {
        this.scene.start('WinScene', { score: newTotal });
      } else {
        this.scene.start('GameScene', { round: this.roundIdx + 1, score: newTotal });
      }
    });
  }
}

// ───────────────────────────────────────────────────────────────
//  WIN SCENE — all rounds complete
// ───────────────────────────────────────────────────────────────
class WinScene extends Phaser.Scene {
  constructor() { super('WinScene'); }
  init(data) { this.finalScore = data.score || 0; }

  create() {
    // Background
    for (let x = 0; x < GW; x += 256)
      this.add.image(x, GH / 2, 'bg_blue').setOrigin(0, 0.5).setDisplaySize(256, GH);
    for (let x = 0; x < GW; x += 180)
      this.add.image(x, 0, 'curtain_top').setOrigin(0, 0).setDisplaySize(180, 58);
    for (let x = 0; x < GW; x += 200)
      this.add.image(x, GH, 'grass1').setOrigin(0, 1).setDisplaySize(200, 110);

    // Random star confetti
    for (let i = 0; i < 22; i++) {
      this.add.text(
        20 + Math.random() * (GW - 40),
        20 + Math.random() * (GH - 40),
        '⭐', { fontSize: `${12 + Math.random() * 22}px` }
      );
    }

    this.add.image(GW / 2, 78, 'txt_gameover').setScale(1.35);

    this.add.text(GW / 2, 162, `All ${ROUNDS.length} rounds complete!`, {
      fontFamily: 'Arial Black', fontSize: '26px', color: '#FFFFFF'
    }).setOrigin(0.5);

    this.add.text(GW / 2, 218, `Final Score:  ${this.finalScore}`, {
      fontFamily: 'Arial Black', fontSize: '34px', color: '#FFD700'
    }).setOrigin(0.5);

    // Star rating
    const maxScore = ROUNDS.length * MAX_AMMO * 100;
    const pct = this.finalScore / maxScore;
    this.add.text(GW / 2, 282,
      pct > 0.75 ? '⭐⭐⭐' : pct > 0.45 ? '⭐⭐' : '⭐',
      { fontSize: '50px' }
    ).setOrigin(0.5);

    // Bouncing ducks
    ['duck_yellow', 'duck_brown', 'duck_white'].forEach((k, i) => {
      const img = this.add.image(248 + i * 202, 385, k).setScale(1.1);
      this.tweens.add({ targets: img, y: 370, yoyo: true, repeat: -1,
        duration: 510 + i * 110, ease: 'Sine.InOut' });
    });

    // Play again
    const btn = this.add.rectangle(GW / 2, 476, 260, 62, 0xFF7700)
      .setInteractive({ useHandCursor: true });
    this.add.text(GW / 2, 476, '🔄  PLAY AGAIN', {
      fontFamily: 'Arial Black', fontSize: '22px', color: '#ffffff'
    }).setOrigin(0.5);
    btn.on('pointerover', () => btn.setFillStyle(0xFF9933));
    btn.on('pointerout',  () => btn.setFillStyle(0xFF7700));
    btn.on('pointerdown', () => this.scene.start('GameScene', { round: 0, score: 0 }));

    // Menu
    const mb = this.add.rectangle(GW / 2, 548, 182, 44, 0x444444)
      .setInteractive({ useHandCursor: true });
    this.add.text(GW / 2, 548, '⬅  Main Menu', {
      fontFamily: 'Arial', fontSize: '16px', color: '#ffffff'
    }).setOrigin(0.5);
    mb.on('pointerdown', () => this.scene.start('MenuScene'));
  }
}

// ───────────────────────────────────────────────────────────────
//  PHASER CONFIG
// ───────────────────────────────────────────────────────────────
const config = {
  type:            Phaser.AUTO,
  width:           GW,
  height:          GH,
  parent:          'game-container',
  backgroundColor: '#0a0a1a',
  scene:           [BootScene, MenuScene, GameScene, WinScene],
};

new Phaser.Game(config);
