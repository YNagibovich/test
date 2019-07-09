TARGET   = program

#MODE=DEBUG
MODE=RELEASE

CUDAINC = /usr/local/cuda-10.1/targets/x86_64-linux/include
CUDALIB = /usr/local/cuda-10.1/targets/x86_64-linux/lib


BUILDDATE := $(shell date +%d%m%Y)

ifeq ($(MODE),DEBUG)
CFLAGS   = -std=c++11 -Wall -I./include -I$(CUDAINC) -g3 -ggdb -DDEBUG -DBUILDDATE=\"$(BUILDDATE)\" -DNOCUDA
else
CFLAGS   = -std=c++11 -Wall -I./include -I$(CUDAINC) -O1 -ggdb -DNDEBUG -DBUILDDATE=\"$(BUILDDATE)\" -DNOCUDA
endif

CC       = gcc

LINKER   = gcc

LFLAGS   = -Wall -I. -lm -lstdc++ -L./lib/ -L$(CUDALIB) -ldl -lturbojpeg -lpthread -lcudart

SRCDIR   = src
OBJDIR   = obj
BINDIR   = bin

SOURCES  := $(wildcard $(SRCDIR)/*.cpp)
INCLUDES := $(wildcard $(SRCDIR)/*.h)
OBJECTS  := $(SOURCES:$(SRCDIR)/%.cpp=$(OBJDIR)/%.o)
rm       = rm -f

$(BINDIR)/$(TARGET):$(OBJECTS)
	@$(LINKER) $(OBJECTS) $(LFLAGS) -o $@
	@echo "Linking complete!"

$(OBJECTS): $(OBJDIR)/%.o : $(SRCDIR)/%.cpp
	@$(CC) $(CFLAGS) -c $< -o $@
	@echo "Compiled "$<" successfully!"

clean:
	@$(rm) $(OBJECTS)
	@echo "Cleanup complete!"

.PHONY: all
remove: clean
	@$(rm) $(BINDIR)/$(TARGET)
	@echo "Executable removed!"