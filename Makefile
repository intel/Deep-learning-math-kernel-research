CXX      = icc
CXXFLAGS = -g -O2 -std=c++11 -fopenmp -qopt-report=5 -Iinclude/ -Isrc/
LDFLAGS  = -Llib/
LDLIBS   = -liomp5

SRCDIR   := src
OBJDIR   := build
SOURCES  := $(shell ls src/*.cpp)
OBJECTS  := $(SOURCES:$(SRCDIR)/%.cpp=$(OBJDIR)/%.o)

lib      = libel.so
test     = $(OBJDIR)/conv

all: $(test)

$(test): $(OBJECTS)
	@mkdir -p $(OBJDIR)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $(test) $(OBJECTS) $(LDLIBS)

$(OBJECTS): $(OBJDIR)/%.o : $(SRCDIR)/%.cpp
	$(CXX) $(CXXFLAGS) -o $@ -c $<

clean:
	rm -f $(OBJECTS) $(OBJDIR)/*.optrpt $(test) $(lib)

