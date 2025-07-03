package database

import (
	"fmt"
	"sync"

	"gorm.io/driver/sqlite"
	"gorm.io/gorm"
	"gorm.io/gorm/logger"
)

// Manager handles database connection and initialization
type Manager struct {
	db       *gorm.DB
	initLock sync.Mutex
}

// NewManager creates a new database manager
func NewManager(databasePath string) (*Manager, error) {
	db, err := gorm.Open(sqlite.Open(databasePath), &gorm.Config{
		Logger: logger.Default.LogMode(logger.Info),
	})
	if err != nil {
		return nil, fmt.Errorf("failed to connect to database: %w", err)
	}

	// Enable foreign key constraints for SQLite
	sqlDB, err := db.DB()
	if err != nil {
		return nil, fmt.Errorf("failed to get underlying sql.DB: %w", err)
	}

	if _, err := sqlDB.Exec("PRAGMA foreign_keys = ON"); err != nil {
		return nil, fmt.Errorf("failed to enable foreign key constraints: %w", err)
	}

	return &Manager{db: db}, nil
}

// Initialize sets up the database tables
func (m *Manager) Initialize() error {
	if !m.initLock.TryLock() {
		return fmt.Errorf("database initialization already in progress")
	}
	defer m.initLock.Unlock()

	// AutoMigrate all models
	err := m.db.AutoMigrate(
		&Agent{},
		&Session{},
		&Task{},
		&Message{},
		&PushNotification{},
		&Feedback{},
		&Tool{},
		&ToolServer{},
		&EvalTask{},
		&EvalCriteria{},
		&EvalRun{},
	)

	if err != nil {
		return fmt.Errorf("failed to migrate database: %w", err)
	}

	return nil
}

// Reset drops all tables and optionally recreates them
func (m *Manager) Reset(recreateTables bool) error {
	if !m.initLock.TryLock() {
		return fmt.Errorf("database reset already in progress")
	}
	defer m.initLock.Unlock()

	// Drop all tables
	err := m.db.Migrator().DropTable(
		&Agent{},
		&Session{},
		&Task{},
		&Message{},
		&PushNotification{},
		&Feedback{},
		&Tool{},
		&ToolServer{},
		&EvalTask{},
		&EvalCriteria{},
		&EvalRun{},
	)

	if err != nil {
		return fmt.Errorf("failed to drop tables: %w", err)
	}

	if recreateTables {
		return m.Initialize()
	}

	return nil
}

// Close closes the database connection
func (m *Manager) Close() error {
	sqlDB, err := m.db.DB()
	if err != nil {
		return err
	}
	return sqlDB.Close()
}
