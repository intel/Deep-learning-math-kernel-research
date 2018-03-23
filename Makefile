CXX      = icc
DFLAGS   = -g #-qopt-report=5
CXXFLAGS = -O2 -std=c++11 -fopenmp -Iinclude/ -Isrc/ $(DFLAGS)
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
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $(test) $(OBJECTS) $(LDLIBS)

$(OBJECTS): $(OBJDIR)/%.o : $(SRCDIR)/%.cpp | $(OBJDIR)
	$(CXX) $(CXXFLAGS) -o $@ -c $<

$(OBJDIR):
	@mkdir -p $(OBJDIR)
    
clean:
	rm -f $(OBJECTS) $(OBJDIR)/*.optrpt $(test) $(lib)

