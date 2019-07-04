TARGET   = program

#MODE=DEBUG
MODE=RELEASE

BUILDDATE := $(shell date +%d%m%Y)

ifeq ($(MODE),DEBUG)
CFLAGS   = -std=c++11 -Wall -I./include -g3 -ggdb -DDEBUG -DBUILDDATE=\"$(BUILDDATE)\" 
else
CFLAGS   = -std=c++11 -Wall -I./include -O1 -ggdb -DNDEBUG -DBUILDDATE=\"$(BUILDDATE)\"
endif

CC       = gcc

LINKER   = gcc

LFLAGS   = -Wall -I. -lm -lstdc++ -L./lib/ -ldl -lturbojpeg -lpthread

SRCDIR   = src
OBJDIR   = obj
BINDIR   = bin

SOURCES  := $(wildcard $(SRCDIR)/*.cpp)
INCLUDES := $(wildcard $(SRCDIR)/*.h)
OBJECTS  := $(SOURCES:$(SRCDIR)/%.cpp=$(OBJDIR)/%.o)
rm       = rm -f

$(BINDIR)/$(TARGET): $(OBJECTS)
	@$(LINKER) $(OBJECTS) $(LFLAGS) -o $@
	@echo "Linking complete!"

$(OBJECTS): $(OBJDIR)/%.o : $(SRCDIR)/%.cpp
	@$(CC) $(CFLAGS) -c $< -o $@
	@echo "Compiled "$<" successfully!"

.PHONY: clean
clean:
	@$(rm) $(OBJECTS)
	@echo "Cleanup complete!"

.PHONY: remove
remove: clean
	@$(rm) $(BINDIR)/$(TARGET)
	@echo "Executable removed!"