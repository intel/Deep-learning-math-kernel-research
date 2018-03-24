CXX      := icc
DFLAGS   := -g #-qopt-report=5
CXXFLAGS := -O2 -std=c++11 -fopenmp -fPIC -Iinclude/ -Isrc/ $(DFLAGS)
LDFLAGS  := -Llib/
LDLIBS   := -liomp5

SRC_DIR  := src
OBJ_DIR  := build
TEST_DIR := test
BIN_DIR  := $(OBJ_DIR)/bin
LIB_DIR  := $(OBJ_DIR)/lib

TCXX     := icc
TCXXFLAGS:= -g -O2 -std=c++11 -fopenmp -pie -fPIE -Iinclude/
TLDFLAGS := -Llib -L$(LIB_DIR) -liomp5 -lel -Wl,-rpath=$(LIB_DIR)

SOURCES  := $(shell ls $(SRC_DIR)/*.cpp)
TESTS    := $(shell ls $(TEST_DIR)/*.cpp)
OBJECTS  := $(SOURCES:$(SRC_DIR)/%.cpp=$(OBJ_DIR)/$(SRC_DIR)/%.o)
TESTOBJS := $(TESTS:$(TEST_DIR)/%.cpp=$(OBJ_DIR)/$(TEST_DIR)/%.o)

LIB      = $(LIB_DIR)/libel.so
TEST     = $(BIN_DIR)/el_conv

.PHONY: lib test clean

all  : lib test
lib  : $(LIB)
test : $(TEST)

$(TEST): $(TESTOBJS) $(LIB)
	@mkdir -p $(dir $@)
	$(TCXX) $(TCXXFLAGS) $(TLDFLAGS) -o $@ $(TESTOBJS)

$(TESTOBJS): $(OBJ_DIR)/$(TEST_DIR)/%.o : $(TEST_DIR)/%.cpp
	@mkdir -p $(dir $@)
	$(TCXX) $(TCXXFLAGS) -o $@ -c $<

$(LIB): $(OBJECTS)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) --shared -o $@ $(OBJECTS) $(LDLIBS)

$(OBJECTS): $(OBJ_DIR)/$(SRC_DIR)/%.o : $(SRC_DIR)/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -o $@ -c $<

clean:
	@rm -f $(TEST) $(LIB)
	@find $(OBJ_DIR) -name "*.o" -exec rm -f {} \;

