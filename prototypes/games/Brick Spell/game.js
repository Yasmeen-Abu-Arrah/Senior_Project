'use strict';

// ═══════════════════════════════════════════════════════════════
//  BRICK SPELL — Word-Building Game for Communication Disorders
//  Uses Kenney Toy Brick Pack assets
//
//  Gameplay:
//    • An emoji picture and a hint clue appear on screen
//    • Coloured letter bricks are shown in scrambled order
//      🟡 Yellow = vowel (A E I O U)
//      🔵 Blue   = consonant
//      🔴 Red    = distractor (not in the word)
//    • Click letters IN ORDER to spell the target word
//    • First-try bonus: +50 pts  |  Each correct letter: +10 pts
//    • An emotion character reacts on word completion
// ═══════════════════════════════════════════════════════════════

const GW   = 800;
const GH   = 600;
const BASE = '../New folder/kenney_toy-brick-pack/PNG/Default';

const VOWELS = new Set(['A','E','I','O','U']);

// 12 words across 3 difficulty tiers (3-letter → 4-letter → 5-letter)
const WORDS = [
  { word:'CAT',   emoji:'🐱', hint:'A fluffy pet that says "meow"'       },
  { word:'DOG',   emoji:'🐶', hint:'A loyal pet that says "woof"'         },
  { word:'SUN',   emoji:'☀️', hint:'It shines bright in the sky'          },
  { word:'HAT',   emoji:'🎩', hint:'You wear it on your head'             },
  { word:'FISH',  emoji:'🐟', hint:'It swims in the water'                },
  { word:'BIRD',  emoji:'🐦', hint:'It has wings and can fly'             },
  { word:'CAKE',  emoji:'🎂', hint:'A sweet birthday treat'               },
  { word:'MILK',  emoji:'🥛', hint:'A healthy white drink'                },
  { word:'APPLE', emoji:'🍎', hint:'A round red or green fruit'           },
  { word:'SMILE', emoji:'😊', hint:'Your happy face expression'           },
  { word:'CHAIR', emoji:'🪑', hint:'You sit on this every day'            },
  { word:'HORSE', emoji:'🐴', hint:'A large animal you can ride'          },
];

/** Fisher-Yates shuffle — returns a new shuffled array */
function shuffle(arr) {
  const a = [...arr];
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

// ───────────────────────────────────────────────────────────────
//  BOOT SCENE — asset loading with progress bar
// ───────────────────────────────────────────────────────────────
class BootScene extends Phaser.Scene {
  constructor() { super('BootScene'); }

  preload() {
    // Progress bar
    this.add.rectangle(GW / 2, GH / 2 - 18, 504, 34, 0x333333);
    const bar = this.add.rectangle(GW / 2 - 250, GH / 2 - 18, 0, 30, 0xFFC300)
      .setOrigin(0, 0.5);
    this.add.text(GW / 2, GH / 2 - 55, 'Loading assets…', {
      fontFamily: 'Arial', fontSize: '20px', color: '#ffffff'
    }).setOrigin(0.5);
    this.load.on('progress', v => bar.setDisplaySize(500 * v, 30));

    // Medium 2-stud brick in each colour
    ['Yellow', 'Blue', 'Red', 'Green', 'White'].forEach(c => {
      this.load.image(`brick_${c.toLowerCase()}`,
        `${BASE}/${c}/brick_medium_2.png`);
    });

    // Emotion character bricks  (a=happy  b=sad  c=calm  d=cool  e=excited)
    ['a', 'b', 'c', 'd', 'e'].forEach(k => {
      this.load.image(`char_${k}`, `${BASE}/Special/extra_character_${k}.png`);
    });

    // Special bricks
    this.load.image('box_coin', `${BASE}/Special/extra_box_coin.png`);
    this.load.image('box_bang', `${BASE}/Special/extra_box_exclamation.png`);
  }

  create() { this.scene.start('MenuScene'); }
}

// ───────────────────────────────────────────────────────────────
//  MENU SCENE
// ───────────────────────────────────────────────────────────────
class MenuScene extends Phaser.Scene {
  constructor() { super('MenuScene'); }

  create() {
    // Sky-blue gradient background
    const bg = this.add.graphics();
    bg.fillGradientStyle(0x74B9FF, 0x74B9FF, 0xDFE6E9, 0xDFE6E9, 1);
    bg.fillRect(0, 0, GW, GH);

    // Ground decoration – row of bricks
    const groundPalette = ['yellow','blue','red','green','yellow','blue','red','green','yellow','blue'];
    groundPalette.forEach((c, i) => {
      this.add.image(40 + i * 80, GH - 28, `brick_${c}`).setDisplaySize(76, 42);
    });

    // ── Title ──────────────────────────────────────────────────
    this.add.text(GW / 2, 108, 'BRICK SPELL', {
      fontFamily: 'Impact, Arial Black, sans-serif',
      fontSize:   '74px',
      color:      '#FFC300',
      stroke:     '#A0522D',
      strokeThickness: 7,
      shadow: { offsetX:4, offsetY:4, color:'#00000070', blur:4, fill:true }
    }).setOrigin(0.5);

    this.add.text(GW / 2, 186, 'Build Words  ·  Learn  ·  Have Fun!', {
      fontFamily: 'Arial, sans-serif',
      fontSize:   '20px',
      color:      '#2D3436'
    }).setOrigin(0.5);

    // ── Colour legend ──────────────────────────────────────────
    const legend = [
      ['brick_yellow', 'Vowels  (A E I O U)'],
      ['brick_blue',   'Consonants'],
      ['brick_red',    'Distractor  (not in word)'],
    ];
    legend.forEach(([key, label], i) => {
      const x = 108 + i * 218;
      this.add.image(x, 248, key).setDisplaySize(38, 22);
      this.add.text(x + 26, 248, label, {
        fontFamily: 'Arial', fontSize: '13px', color: '#555555'
      }).setOrigin(0, 0.5);
    });

    // ── Bouncing emotion characters ────────────────────────────
    ['char_a','char_c','char_e','char_d','char_b'].forEach((k, i) => {
      const img = this.add.image(120 + i * 140, 328, k).setDisplaySize(62, 62);
      this.tweens.add({
        targets: img, y: 316,
        yoyo: true, repeat: -1,
        duration: 560 + i * 90,
        ease: 'Sine.InOut'
      });
    });

    // ── Instructions ──────────────────────────────────────────
    this.add.text(GW / 2, 410, [
      '📖  How to play',
      '    An emoji picture appears with a hint clue',
      '    Click the letters in the CORRECT ORDER to spell the word',
      '    Spell all 12 words to win!  •  First-try bonus: +50 pts'
    ].join('\n'), {
      fontFamily: 'Arial', fontSize: '14px',
      color: '#444444', lineSpacing: 5
    }).setOrigin(0.5);

    // ── Play button ────────────────────────────────────────────
    const btn = this.add.rectangle(GW / 2, 508, 260, 62, 0x00B894)
      .setInteractive({ useHandCursor: true });
    this.add.text(GW / 2, 508, '▶  START PLAYING', {
      fontFamily: 'Arial Black', fontSize: '22px', color: '#ffffff'
    }).setOrigin(0.5);

    btn.on('pointerover', () => btn.setFillStyle(0x55EFC4));
    btn.on('pointerout',  () => btn.setFillStyle(0x00B894));
    btn.on('pointerdown', () => {
      this.scene.start('GameScene', { wordIndex: 0, score: 0 });
    });

    // Word count note
    this.add.text(GW / 2, 560, `${WORDS.length} words across 3 difficulty levels`, {
      fontFamily: 'Arial', fontSize: '13px', color: '#888888'
    }).setOrigin(0.5);
  }
}

// ───────────────────────────────────────────────────────────────
//  GAME SCENE — one word per visit
// ───────────────────────────────────────────────────────────────
class GameScene extends Phaser.Scene {
  constructor() { super('GameScene'); }

  init(data) {
    this.wordIndex   = data.wordIndex ?? 0;
    this.score       = data.score     ?? 0;
    this.firstTry    = true;
    this.wordDone    = false;
    this.nextSlotIdx = 0;
  }

  create() {
    const data = WORDS[this.wordIndex % WORDS.length];
    this.wordData   = data;
    this.targetWord = data.word;

    // ── Background ─────────────────────────────────────────────
    const bg = this.add.graphics();
    bg.fillGradientStyle(0xFEF9E7, 0xFEF9E7, 0xFFF3CD, 0xFFF3CD, 1);
    bg.fillRect(0, 0, GW, GH);

    // ── Top bar ────────────────────────────────────────────────
    this.add.rectangle(GW / 2, 30, GW, 60, 0x2C3E50);

    this.scoreTxt = this.add.text(18, 30, `⭐ ${this.score}`, {
      fontFamily: 'Arial Black', fontSize: '22px', color: '#FFD700'
    }).setOrigin(0, 0.5);

    const wordNum = (this.wordIndex % WORDS.length) + 1;
    this.add.text(GW / 2, 22, `Word  ${wordNum}  /  ${WORDS.length}`, {
      fontFamily: 'Arial', fontSize: '15px', color: '#ECF0F1'
    }).setOrigin(0.5);

    // Difficulty label
    const tier = wordNum <= 4 ? 'Level 1 — 3 letters' :
                 wordNum <= 8 ? 'Level 2 — 4 letters' : 'Level 3 — 5 letters';
    this.add.text(GW / 2, 40, tier, {
      fontFamily: 'Arial', fontSize: '12px', color: '#95A5A6'
    }).setOrigin(0.5);

    this.add.text(GW - 14, 30, `💡 ${data.hint}`, {
      fontFamily: 'Arial', fontSize: '13px', color: '#F39C12',
      wordWrap: { width: 290 }
    }).setOrigin(1, 0.5);

    // ── Emoji picture ──────────────────────────────────────────
    this.add.text(GW / 2, 140, data.emoji, { fontSize: '90px' }).setOrigin(0.5);

    this.add.text(GW / 2, 212, 'Click the letters in the right order ↓', {
      fontFamily: 'Arial', fontSize: '16px', color: '#7F8C8D'
    }).setOrigin(0.5);

    // ── Build zone (blank slots) ───────────────────────────────
    this.slots = this._createSlots();

    // ── Letter bricks ─────────────────────────────────────────
    this.bricks = this._createBricks();

    // ── Feedback character (slides in on completion) ───────────
    this.feedbackChar = this.add.image(GW + 80, 430, 'char_a')
      .setDisplaySize(72, 72)
      .setAlpha(0);
  }

  // ── Build zone ──────────────────────────────────────────────
  _createSlots() {
    const n   = this.targetWord.length;
    const sw  = 72, gap = 10;
    const total = n * sw + (n - 1) * gap;
    const sx    = (GW - total) / 2 + sw / 2;
    const y     = 300;

    return Array.from({ length: n }, (_, i) => {
      const x = sx + i * (sw + gap);
      const brick  = this.add.image(x, y, 'brick_white').setDisplaySize(sw, 50);
      const letter = this.add.text(x, y, '_', {
        fontFamily: 'Arial Black', fontSize: '26px', color: '#BDC3C7'
      }).setOrigin(0.5);
      return { x, y, brick, letter };
    });
  }

  // ── Letter bricks (word letters + distractors, shuffled) ────
  _createBricks() {
    const wordLetters = this.targetWord.split('');
    const usedSet     = new Set(wordLetters);

    // Pick distractor letters not present in the word
    const pool  = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'.split('').filter(l => !usedSet.has(l));
    const numD  = Math.min(4, Math.max(2, 8 - wordLetters.length));
    const all   = shuffle([...wordLetters, ...shuffle(pool).slice(0, numD)]);

    const bw    = 72, gap = 8;
    const total = all.length * bw + (all.length - 1) * gap;
    const sx    = (GW - total) / 2 + bw / 2;
    const y     = 440;

    return all.map((letter, i) => {
      const x        = sx + i * (bw + gap);
      const isD      = !usedSet.has(letter);
      const brickKey = isD ? 'brick_red'
                     : VOWELS.has(letter) ? 'brick_yellow' : 'brick_blue';

      const img  = this.add.image(x, y, brickKey).setDisplaySize(bw, 50);
      const txt  = this.add.text(x, y - 2, letter, {
        fontFamily: 'Arial Black', fontSize: '26px',
        color: '#ffffff',
        stroke: '#00000050', strokeThickness: 3
      }).setOrigin(0.5);

      const zone = this.add.zone(x, y, bw, 64).setInteractive({ useHandCursor: true });
      const brick = { letter, img, txt, zone, used: false };

      zone.on('pointerover', () => {
        if (!brick.used) { img.setDisplaySize(bw + 8, 54); txt.setScale(1.1); }
      });
      zone.on('pointerout', () => {
        if (!brick.used) { img.setDisplaySize(bw, 50); txt.setScale(1); }
      });
      zone.on('pointerdown', () => {
        if (!this.wordDone && !brick.used) this._onLetterClick(brick);
      });

      return brick;
    });
  }

  // ── Letter click handler ─────────────────────────────────────
  _onLetterClick(brick) {
    const expected = this.targetWord[this.nextSlotIdx];

    if (brick.letter === expected) {
      // ✅ Correct
      brick.used = true;
      brick.zone.removeInteractive();

      const slot = this.slots[this.nextSlotIdx++];
      this.score += 10;
      this.scoreTxt.setText(`⭐ ${this.score}`);

      // Fly brick up to fill the slot
      this.tweens.add({
        targets:  [brick.img, brick.txt],
        x:        slot.x,
        y:        slot.y,
        duration: 320,
        ease:     'Back.Out',
        onComplete: () => {
          slot.brick.setTexture('brick_green');
          slot.letter.setText(brick.letter).setColor('#ffffff');
          brick.img.setAlpha(0);
          brick.txt.setAlpha(0);
        }
      });

      // Green camera flash
      this.cameras.main.flash(80, 60, 210, 80);

      if (this.nextSlotIdx >= this.targetWord.length) {
        this.time.delayedCall(460, () => this._onWordComplete());
      }

    } else {
      // ❌ Wrong — shake + red flash
      this.firstTry = false;
      brick.img.setTint(0xFF4444);
      const ox = brick.img.x;
      const ot = brick.txt.x;

      this.tweens.add({
        targets:  [brick.img, brick.txt],
        x:        ox + 11,
        yoyo:     true,
        repeat:   3,
        duration: 55,
        ease:     'Linear',
        onComplete: () => {
          brick.img.clearTint();
          brick.img.x = ox;
          brick.txt.x = ot;
        }
      });

      this.cameras.main.flash(110, 255, 50, 50);
    }
  }

  // ── Word completed ───────────────────────────────────────────
  _onWordComplete() {
    this.wordDone = true;

    if (this.firstTry) {
      this.score += 50;
      this.scoreTxt.setText(`⭐ ${this.score}`);
    }

    // Slide in the excited character from the right
    this.feedbackChar.setTexture('char_e').setDisplaySize(72, 72).setAlpha(1);
    this.tweens.add({
      targets:  this.feedbackChar,
      x:        GW - 56,
      duration: 480,
      ease:     'Back.Out'
    });

    // Congratulations message
    const msg = this.firstTry ? '🎉  Perfect!  +50 Bonus!' : '✅  Great job!';
    const txt = this.add.text(GW / 2, 530, msg, {
      fontFamily: 'Arial Black', fontSize: '26px',
      color: '#FFD700', stroke: '#E67E22', strokeThickness: 4
    }).setOrigin(0.5).setScale(0);

    this.tweens.add({
      targets: txt, scaleX: 1, scaleY: 1,
      duration: 360, ease: 'Back.Out'
    });

    // Star burst from the build zone
    this._spawnStars(GW / 2, 300);

    // Show navigation button after a short pause
    this.time.delayedCall(720, () => this._showNextBtn());
  }

  // ── Star burst celebration ────────────────────────────────────
  _spawnStars(cx, cy) {
    for (let i = 0; i < 14; i++) {
      const star  = this.add.text(cx, cy, '⭐', { fontSize: '20px' });
      const angle = (Math.PI * 2 / 14) * i;
      const r     = 85 + Math.random() * 35;
      this.tweens.add({
        targets:  star,
        x:        cx + Math.cos(angle) * r,
        y:        cy + Math.sin(angle) * r - 35,
        alpha:    0,
        scaleX:   0,
        scaleY:   0,
        duration: 680 + Math.random() * 320,
        ease:     'Power2',
        onComplete: () => star.destroy()
      });
    }
  }

  // ── Next / Finish button ──────────────────────────────────────
  _showNextBtn() {
    const isLast = (this.wordIndex + 1) >= WORDS.length;
    const label  = isLast ? '🏆  FINISH!' : 'NEXT WORD  ▶';

    const btn = this.add.rectangle(GW / 2, 568, 220, 50, 0x27AE60)
      .setInteractive({ useHandCursor: true });
    this.add.text(GW / 2, 568, label, {
      fontFamily: 'Arial Black', fontSize: '18px', color: '#ffffff'
    }).setOrigin(0.5);

    btn.on('pointerover', () => btn.setFillStyle(0x2ECC71));
    btn.on('pointerout',  () => btn.setFillStyle(0x27AE60));
    btn.on('pointerdown', () => {
      if (isLast) {
        this.scene.start('WinScene', { score: this.score });
      } else {
        this.scene.start('GameScene', {
          wordIndex: this.wordIndex + 1,
          score:     this.score
        });
      }
    });
  }
}

// ───────────────────────────────────────────────────────────────
//  WIN SCENE — all words completed
// ───────────────────────────────────────────────────────────────
class WinScene extends Phaser.Scene {
  constructor() { super('WinScene'); }

  init(data) { this.finalScore = data.score || 0; }

  create() {
    // Purple celebration gradient
    const bg = this.add.graphics();
    bg.fillGradientStyle(0x6C5CE7, 0x6C5CE7, 0xA29BFE, 0xA29BFE, 1);
    bg.fillRect(0, 0, GW, GH);

    // Scattered stars
    for (let i = 0; i < 26; i++) {
      this.add.text(
        20 + Math.random() * (GW - 40),
        20 + Math.random() * (GH - 40),
        '⭐',
        { fontSize: `${13 + Math.random() * 22}px` }
      );
    }

    this.add.text(GW / 2, 90, '🏆  YOU WIN!  🏆', {
      fontFamily: 'Impact, Arial Black, sans-serif',
      fontSize:   '62px',
      color:      '#FFD700',
      stroke:     '#5D3A00',
      strokeThickness: 6
    }).setOrigin(0.5);

    this.add.text(GW / 2, 172, 'All 12 words spelled!', {
      fontFamily: 'Arial Black', fontSize: '28px', color: '#ffffff'
    }).setOrigin(0.5);

    this.add.text(GW / 2, 232, `Final Score:  ${this.finalScore}`, {
      fontFamily: 'Arial Black', fontSize: '34px', color: '#FFD700'
    }).setOrigin(0.5);

    // Star rating  (rough max: 12 × 60 = 720)
    const pct   = this.finalScore / (WORDS.length * 60);
    const stars = pct > 0.85 ? '⭐⭐⭐' : pct > 0.55 ? '⭐⭐' : '⭐';
    this.add.text(GW / 2, 296, stars, { fontSize: '52px' }).setOrigin(0.5);

    // All 5 emotion characters bouncing
    ['char_a','char_c','char_e','char_d','char_b'].forEach((k, i) => {
      const img = this.add.image(100 + i * 150, 400, k).setDisplaySize(68, 68);
      this.tweens.add({
        targets: img, y: 386,
        yoyo: true, repeat: -1,
        duration: 500 + i * 80,
        ease: 'Sine.InOut'
      });
    });

    // Play-again button
    const playBtn = this.add.rectangle(GW / 2, 498, 250, 60, 0xFFC300)
      .setInteractive({ useHandCursor: true });
    this.add.text(GW / 2, 498, '🔄  PLAY AGAIN', {
      fontFamily: 'Arial Black', fontSize: '22px', color: '#2D3436'
    }).setOrigin(0.5);
    playBtn.on('pointerover', () => playBtn.setFillStyle(0xFFDD57));
    playBtn.on('pointerout',  () => playBtn.setFillStyle(0xFFC300));
    playBtn.on('pointerdown', () => {
      this.scene.start('GameScene', { wordIndex: 0, score: 0 });
    });

    // Main menu button
    const menuBtn = this.add.rectangle(GW / 2, 568, 180, 42, 0x636E72)
      .setInteractive({ useHandCursor: true });
    this.add.text(GW / 2, 568, '⬅  Main Menu', {
      fontFamily: 'Arial', fontSize: '16px', color: '#ffffff'
    }).setOrigin(0.5);
    menuBtn.on('pointerover', () => menuBtn.setFillStyle(0x7F8C8D));
    menuBtn.on('pointerout',  () => menuBtn.setFillStyle(0x636E72));
    menuBtn.on('pointerdown', () => this.scene.start('MenuScene'));
  }
}

// ───────────────────────────────────────────────────────────────
//  PHASER GAME CONFIGURATION
// ───────────────────────────────────────────────────────────────
const config = {
  type:            Phaser.AUTO,
  width:           GW,
  height:          GH,
  parent:          'game-container',
  backgroundColor: '#1a1a2e',
  scene:           [BootScene, MenuScene, GameScene, WinScene],
};

new Phaser.Game(config);
