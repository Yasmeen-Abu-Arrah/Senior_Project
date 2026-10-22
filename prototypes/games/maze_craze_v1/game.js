// --- CONSTANTS & DATA GRID ---
const TILE_SIZE = 40;

// Explicit 2D Array Matrix (0=Path, 1=Wall, 2=Coin, 3=Goal)
const mazeGrid = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 1, 0, 0, 0, 0, 2, 2, 1],
    [1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1],
    [1, 2, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1],
    [1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1],
    [1, 0, 0, 2, 0, 1, 0, 0, 0, 1, 0, 1],
    [1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1],
    [1, 2, 0, 1, 0, 0, 0, 1, 2, 0, 3, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
];

// --- SCENE 1: CHARACTER SELECTION ---
class CharacterSelectScene extends Phaser.Scene {
    constructor() {
        super('CharacterSelectScene');
    }

    preload() {
        this.createSquareTexture('hero_blue', 0x3498db);
        this.createSquareTexture('hero_red', 0xe74c3c);
        this.createSquareTexture('hero_green', 0x2ecc71);
    }

    create() {
        this.add.text(60, 60, 'SELECT YOUR CHARACTER', { fontSize: '24px', fill: '#ffffff', fontStyle: 'bold' });
        this.add.text(60, 100, 'Click a square below to choose:', { fontSize: '14px', fill: '#cccccc' });

        const options = ['hero_blue', 'hero_red', 'hero_green'];
        const names = ['Aqua Knight', 'Crimson Rogue', 'Jade Ranger'];
        this.selectedKey = 'hero_blue';
        
        this.selectedIndicator = this.add.graphics();
        this.selectedIndicator.lineStyle(3, 0xffffff);
        this.selectedIndicator.strokeRect(120 - 30, 200 - 30, 60, 60);

        options.forEach((key, index) => {
            let xPos = 120 + (index * 120);
            let yPos = 200;

            let character = this.add.sprite(xPos, yPos, key).setInteractive().setScale(1.5);
            this.add.text(xPos - 40, yPos + 45, names[index], { fontSize: '12px', fill: '#ffffff', fontStyle: 'bold' });

            character.on('pointerdown', () => {
                this.selectedKey = key;
                this.selectedIndicator.clear();
                this.selectedIndicator.lineStyle(3, 0xffffff);
                this.selectedIndicator.strokeRect(xPos - 30, yPos - 30, 60, 60);
            });
        });

        let startBtn = this.add.rectangle(240, 320, 220, 50, 0x27ae60).setInteractive();
        this.add.text(185, 310, 'LAUNCH GAME', { fontSize: '18px', fill: '#ffffff', fontStyle: 'bold' });

        startBtn.on('pointerover', () => startBtn.setFillStyle(0x2ecc71));
        startBtn.on('pointerout', () => startBtn.setFillStyle(0x27ae60));
        
        startBtn.on('pointerdown', () => {
            this.registry.set('chosenCharacterKey', this.selectedKey);
            this.scene.start('MazeScene');
        });
    }

    createSquareTexture(key, color) {
        let graphics = this.make.graphics({ x: 0, y: 0, add: false });
        graphics.fillStyle(color, 1);
        graphics.fillRect(0, 0, TILE_SIZE - 10, TILE_SIZE - 10);
        graphics.generateTexture(key, TILE_SIZE - 10, TILE_SIZE - 10);
    }
}

// --- SCENE 2: MAZE GAME ---
class MazeScene extends Phaser.Scene {
    constructor() {
        super('MazeScene');
    }

    preload() {
        // --- LOAD ALL GAME AUDIO ASSETS ---
        this.load.audio('coin_sound', 'assets/coin-recieved.mp3');
        this.load.audio('die_sound', 'assets/game-die.mp3');
        this.load.audio('victory_sound', 'assets/gaming-victory.mp3');

        // Visual Generation Graphics 
        let wallGraphics = this.make.graphics({ x: 0, y: 0, add: false });
        wallGraphics.fillStyle(0x7f8c8d, 1);
        wallGraphics.fillRect(0, 0, TILE_SIZE, TILE_SIZE);
        wallGraphics.lineStyle(2, 0x34495e);
        wallGraphics.strokeRect(0, 0, TILE_SIZE, TILE_SIZE);
        wallGraphics.generateTexture('wall_tile', TILE_SIZE, TILE_SIZE);

        let coinGraphics = this.make.graphics({ x: 0, y: 0, add: false });
        coinGraphics.fillStyle(0xf1c40f, 1);
        coinGraphics.fillCircle(TILE_SIZE/2, TILE_SIZE/2, TILE_SIZE/4);
        coinGraphics.generateTexture('coin_tile', TILE_SIZE, TILE_SIZE);

        let goalGraphics = this.make.graphics({ x: 0, y: 0, add: false });
        goalGraphics.fillStyle(0x9b59b6, 1);
        goalGraphics.fillRect(TILE_SIZE/4, TILE_SIZE/4, TILE_SIZE/2, TILE_SIZE/2);
        goalGraphics.generateTexture('goal_tile', TILE_SIZE, TILE_SIZE);

        let enemyGraphics = this.make.graphics({ x: 0, y: 0, add: false });
        enemyGraphics.fillStyle(0xd63031, 1);
        enemyGraphics.fillTriangle(TILE_SIZE/2, 4, 4, TILE_SIZE - 4, TILE_SIZE - 4, TILE_SIZE - 4);
        enemyGraphics.generateTexture('enemy_tile', TILE_SIZE, TILE_SIZE);
    }

    create() {
        this.score = 0;
        this.isGameOver = false;
        this.scoreText = this.add.text(20, 10, 'Coins Collected: 0', { fontSize: '16px', fill: '#ffffff', fontStyle: 'bold' });
        this.inputDebugText = this.add.text(240, 10, 'Press Arrow Keys To Move', { fontSize: '14px', fill: '#f39c12' });

        this.walls = this.physics.add.staticGroup();
        this.coins = this.physics.add.group();
        this.goals = this.physics.add.staticGroup();

        mazeGrid.forEach((row, y) => {
            row.forEach((value, x) => {
                let posX = x * TILE_SIZE + TILE_SIZE / 2;
                let posY = y * TILE_SIZE + TILE_SIZE / 2 + 40; 

                if (value === 1) {
                    this.walls.create(posX, posY, 'wall_tile');
                } else if (value === 2) {
                    this.coins.create(posX, posY, 'coin_tile');
                } else if (value === 3) {
                    this.goals.create(posX, posY, 'goal_tile');
                }
            });
        });

        const selectedSkin = this.registry.get('chosenCharacterKey') || 'hero_blue';
        this.player = this.physics.add.sprite(1 * TILE_SIZE + TILE_SIZE / 2, 1 * TILE_SIZE + TILE_SIZE / 2 + 40, selectedSkin);
        this.player.gridX = 1;
        this.player.gridY = 1;
        this.isMoving = false;

        this.enemy = this.physics.add.sprite(6 * TILE_SIZE + TILE_SIZE / 2, 3 * TILE_SIZE + TILE_SIZE / 2 + 40, 'enemy_tile');
        this.enemy.gridX = 6;
        this.enemy.gridY = 3;
        this.enemyDirection = 1;

        this.enemyTimer = this.time.addEvent({
            delay: 500,
            callback: this.moveEnemy,
            callbackScope: this,
            loop: true
        });

        this.cursors = this.input.keyboard.createCursorKeys();

        // Player collecting a coin
        this.physics.add.overlap(this.player, this.coins, (player, coin) => {
            coin.destroy();
            this.sound.play('coin_sound');
            this.score += 1;
            this.scoreText.setText('Coins Collected: ' + this.score);
        });

        // --- UPDATED OVERLAP TO PLAY VICTORY AUDIO ---
        this.physics.add.overlap(this.player, this.goals, () => {
            this.sound.play('victory_sound');
            this.endGame('STAGE CLEAR!', '#2ecc71');
        });

        // Player hitting the enemy bot
        this.physics.add.overlap(this.player, this.enemy, () => {
            this.sound.play('die_sound');
            this.endGame('GAME OVER!', '#e74c3c');
        });
    }

    update() {
        if (this.isMoving || this.isGameOver) return;

        if (Phaser.Input.Keyboard.JustDown(this.cursors.left)) {
            this.processMove(-1, 0);
        } else if (Phaser.Input.Keyboard.JustDown(this.cursors.right)) {
            this.processMove(1, 0);
        } else if (Phaser.Input.Keyboard.JustDown(this.cursors.up)) {
            this.processMove(0, -1);
        } else if (Phaser.Input.Keyboard.JustDown(this.cursors.down)) {
            this.processMove(0, 1);
        }
    }

    processMove(deltaX, deltaY) {
        let targetX = this.player.gridX + deltaX;
        let targetY = this.player.gridY + deltaY;

        if (mazeGrid[targetY] !== undefined && mazeGrid[targetY][targetX] !== undefined) {
            let destinationTileType = mazeGrid[targetY][targetX];
            
            if (destinationTileType !== 1) {
                this.isMoving = true;
                this.player.gridX = targetX;
                this.player.gridY = targetY;

                this.tweens.add({
                    targets: this.player,
                    x: targetX * TILE_SIZE + TILE_SIZE / 2,
                    y: targetY * TILE_SIZE + TILE_SIZE / 2 + 40,
                    duration: 150,
                    onComplete: () => {
                        this.isMoving = false;
                    }
                });
            } else {
                this.inputDebugText.setText('Blocked by Wall!');
            }
        }
    }

    moveEnemy() {
        if (this.isGameOver) return;

        let targetX = this.enemy.gridX + this.enemyDirection;
        let targetY = this.enemy.gridY;

        if (mazeGrid[targetY] && mazeGrid[targetY][targetX] !== 1) {
            this.enemy.gridX = targetX;

            this.tweens.add({
                targets: this.enemy,
                x: targetX * TILE_SIZE + TILE_SIZE / 2,
                y: targetY * TILE_SIZE + TILE_SIZE / 2 + 40,
                duration: 200
            });
        } else {
            this.enemyDirection *= -1;
        }
    }

    endGame(message, colorText) {
        this.isGameOver = true;
        this.physics.pause();
        this.enemyTimer.destroy();
        
        this.add.rectangle(240, 200, 300, 80, 0x000000, 0.8);
        this.add.text(140, 185, message, { fontSize: '28px', fill: colorText, fontStyle: 'bold' });
    }
}

// --- CONFIGURATION ---
const config = {
    type: Phaser.AUTO,
    width: 480,
    height: 400,
    parent: 'game-container',
    backgroundColor: '#1a252f',
    physics: {
        default: 'arcade',
        arcade: { debug: false }
    },
    scene: [CharacterSelectScene, MazeScene]
};

const game = new Phaser.Game(config);